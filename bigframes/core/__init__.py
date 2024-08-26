# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
import datetime
import functools
import io
import itertools
import typing
from typing import Iterable, Optional, Sequence, Tuple, final
from collections.abc import Mapping
from types import FunctionType, MethodType
import warnings
from collections.abc import Iterator, Callable
from collections import deque

import google.cloud.bigquery
import pandas
import pyarrow as pa
import pyarrow.feather as pa_feather

# schema lineage
from networkx import DiGraph
from google.cloud.bigquery.schema import SchemaField
import bigframes._config as config
from bigframes.core.bqsql_schema_unnest import BQSchemaLayout

import bigframes.core.compile
import bigframes.core.expression as ex
import bigframes.core.guid
#from bigframes.dataframe import DataFrame
import bigframes.core.join_def as join_def
import bigframes.core.local_data as local_data
import bigframes.core.nodes as nodes
import bigframes.core.blocks as blocks
from bigframes.core.ordering import OrderingExpression
import bigframes.core.ordering as orderings
import bigframes.core.rewrite
import bigframes.core.schema as schemata
import bigframes.core.utils
from bigframes.core.window_spec import WindowSpec
#import bigframes.dataframe
import bigframes.dtypes
import bigframes.operations as ops
import bigframes.operations.aggregations as agg_ops
import bigframes.session._io.bigquery

if typing.TYPE_CHECKING:
    from bigframes.session import Session

ORDER_ID_COLUMN = "bigframes_ordering_id"
PREDICATE_COLUMN = "bigframes_predicate"
options = config.options


# -- schema tracking --
def bfnode_hash(node: nodes.BigFrameNode):
    return node._node_hash
    

class SchemaSource:
    def __init__(self, dag: DiGraph, schema_orig: Tuple[SchemaField, ...], schema_bq: BQSchemaLayout) -> None:
        self.schema_orig = schema_orig
        self.schema_bq = schema_bq
        self.dag = dag

    @property
    def is_valid(self) -> bool:
        """
        Returns True if self._dag is not None, which is the case whenever the ArrayValue's BigFrameNode has a physical_schema attribute.
        Other cases will be handled in the near future.
        """
        return self.dag is not None


class SchemaSourceHandler:
    _base_root_name = "_root_"

    def __init__(self):
        self._sources = {}
        self._order = []

    @property
    def sources(self) -> dict:
        return self._sources

    @property
    def order(self) -> list:
        return self._order

    @staticmethod
    def _tree_from_strings(paths: list[str], struct_separator: str) -> dict:
        root = {}
        for path in paths:
            parts = path.split(struct_separator)
            node = root
            for part in parts:
                node = node.setdefault(part, {})
        return root
    
    @staticmethod
    def bfs(tree: list[str]|dict, separator: str, root: str|None=None) -> Iterator[list]:
        """
        Iteraror function for BQ schema base on Tuple[SchemaField, ...]. Returns layer using breadth first search.
        """
        # bfs on "."-joined strings, the "." is the struct_separator
        # start queue with root key-value pair
        queue = deque([[root]]) if root is not None else deque([([], tree)])

        while queue:
            layer = []
            for _ in range(len(queue)):
                # get current item and traverse its direct succesors
                path, node = queue.popleft()
                assert(isinstance(node, dict))
                #assert(isinstance(path, list))
                for key, child in node.items():
                    # build key string by concatenating with path/ predecessor's name. Separator should be struct_separator here.
                    new_path = path + separator + key if path else key # type: ignore
                    # add item to layer/ current level in tree
                    layer.append(new_path)
                    # append item to queue for breadth first search
                    queue.append((new_path, child)) # type: ignore
            if layer:
                yield layer

    def _init_dag_from_schema(self, dag: DiGraph, schema: BQSchemaLayout, layer_separator: str, struct_separator: str) -> DiGraph:
        root_layer = True
        dag_ret = dag
        bq_schema = self._tree_from_strings(list(schema.map_to_list.keys()), struct_separator)
        for layer in self.bfs(bq_schema, separator=struct_separator):
            for col_name in layer:
                assert(layer_separator not in col_name)
                last_layer = col_name.rsplit(struct_separator, 1)[0] if not root_layer else self._base_root_name
                col_type = schema.map_to_type[col_name]
                # replace struct separator with layer separator, as struct separator must not be used in exploded column names
                col_name = col_name.replace(struct_separator, layer_separator)
                dag_ret.add_node(col_name, node_type=col_type)
                dag_ret.add_edge(last_layer, col_name)
            root_layer = False
        return dag_ret
    
    @staticmethod
    def leafs(dag: DiGraph):
        return [node for node in dag.nodes if dag.out_degree(node) == 0]

    # two identical properties, depending on what meaning you prefer
    def _dag_to_schema(self):
        # layers = bfs_layers(self._dag, self._base_root_name)
        # bfs = bfs_tree(self._dag, self._base_root_name)
        # parent_layer = self._base_root_name
        pass

    def add_source(self, src: nodes.BigFrameNode, layer_separator: str, struct_separator: str) -> None:
        """Adds new SchemaSource for src to self._sources"""
        schema_orig: Tuple[SchemaField, ...] = src.physical_schema if hasattr(src, "physical_schema") else None # type: ignore
        schema = None
        dag = None
        if schema_orig:
            schema = BQSchemaLayout(schema_orig)
            schema.determine_layout(struct_separator)
            dag = DiGraph()
            # ONE common root note as multiple columns can follow
            dag.add_node(self._base_root_name, node_type=self._base_root_name)
            dag = self._init_dag_from_schema(dag, schema, layer_separator, struct_separator)
        source = SchemaSource(dag, schema_orig, schema)
        src_hash = bfnode_hash(src)
        self._sources[src_hash] = source
        self._order.append(src_hash)

    def _value_multiplicities(self, input_dict: dict[str, list[str]]) -> dict[tuple, str]:
        """
        Finds multiplicities of values in a dict and returns an 'inverse' dict with keys as value list.
        Changing the key from list to hashable tuple we can cover two cases in once:
            a) single column explodes into multiple others, such as OneHotEncoding
            b) multiple columns merge into a single one, such as Merge for categories with small amount of samples
        :param dict[str, List[str]] input_dict: dict with keys as column names and values as list of column names
        :return: dict with keys as value list and values as list of column names
        """
        inverted_dict = {}
        for key, value in input_dict.items():

            value_tuple = tuple(sorted(value))
            inverted_dict.setdefault(value_tuple, []).append(key)
        duplicates = {value: keys for value, keys in inverted_dict.items() if len(keys) > 1}
        #TODO: add NodeInfo?
        return duplicates

    def exists(self, src: nodes.BigFrameNode) -> SchemaSource|None:
        """Returns SchemaSource if src exists, else None."""
        return self._sources.get(src, None)



# to guarantee singleton creation, we prohibit inheritnig from the class
@final
class SchemaTrackingContextManager:
    """
    Context manager for schema tracking using command pattern.
    Utilizes a DAG for schema lineage and thus can reconstruct each step of schema changes.
    """
    _default_sep_layers: str = "__"  # make sure it is not a substring of any column name!
    _default_sep_structs: str = "."  # not ot be modified by user
    _is_active = False

    # setup, start schema deduction
    #def __init__(self, data: DataFrame | Series | str | None=None, layer_separator: str | None = None):
    def __init__(self, layer_separator: str | None = None, struct_separator: str | None = None):
        # TODO: change into parameters
        # this needs to be done before getting the schema
        self.sep_layers = layer_separator if layer_separator is not None else SchemaTrackingContextManager._default_sep_layers
        self.set_structs = struct_separator if struct_separator is not None else SchemaTrackingContextManager._default_sep_structs
        self._source_handler = SchemaSourceHandler()
        self.block_start: blocks.Block|None = None
        self.block_end: blocks.Block|None = None
        self._latest_op: dict|Mapping = {}  # latest schema changes
        self._latest_callee: str = ""
        self._op_count = 0

    @property
    def num_nested_commands(self) -> int:
        return self._op_count

    def prev_changes(self) -> tuple[str, dict|Mapping]:
        return ((self._latest_callee, self._latest_op))

    def add_changes(self, hdl: str, changes: dict|Mapping):
        self._latest_callee = hdl
        self._latest_op = changes
        self._op_count += 1

    #@property
    @classmethod
    def active(cls):
        """
        Returns True if context manager is active, ie if we are within a "with" block
        """
        return cls._is_active

    def add_source(self, src: nodes.BigFrameNode) -> None:
        """Adds new SchemaSource for src to self._sources. Key is src."""
        if self._source_handler.exists(src) is not None:
            raise ValueError(f"{self.__class__.__name__}:{self.__class__.__qualname__}: Source {src} already exists")
        self._source_handler.add_source(src, layer_separator=self.sep_layers, struct_separator=self.set_structs)

    def _schemata_matching(self, changes: dict|Mapping) -> bool:
        # compare changes keys to leafs to DAG including nesting
        
        schema_src = src.physical_schema
        #TODO: compare src schema to hdl schema. Otherwise DAG mismatch
        return True
    
    def _extend_dag(self, src: nodes.BigFrameNode, dst: nodes.BigFrameNode) -> None:
        #TODO: extend dag
        return

    def step(self):  #nodes.BigFrameNode):
        assert(self.block_start is not None)
        assert(self.block_end is not None)
        hash_start = bfnode_hash(self.block_start.expr.node)
        hash_parent =bfnode_hash(self.block_end.expr.node.child)
        assert(hash_start==hash_parent)
        hdl = None
        if parent is not None:
            hash_parent = bfnode_hash(parent)
            hdl = self._source_handler.sources.get(hash_parent, None)
        if hdl is None:
            raise ValueError(f"NestedDataCM: Unknown data source {self.block_start}")
        # no join, merge etc., no new source/BigFrameNode
        if not self._schemata_matching(self.block_start, hdl):
            raise Exception("Internal error: Nested Schema mismatch")
            self._extend_dag(hdl, self.block_end.expr.node)

    def reset_block_markers(self):
        self.block_start = None
        self.block_end = None
        return

    # Context Manager interface
    def __enter__(self):
        assert(options.bigquery.project is not None and options.bigquery.location is not None)
        SchemaTrackingContextManager._is_active = True
        self.reset_block_markers()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        #self._dag_to_schema()
        SchemaTrackingContextManager._is_active = False
        #TODO: compute final schema from DAG
        #TODO: delete DAG so "new" context can be used
        #TODO: Get new source in case of joins. new table name/ which one is target?
        return

    # Private helper methods for starting schema deduction and DAG creation
    @staticmethod
    def _has_nested_data(schema: list) -> dict | None:
        return sum([1 for x in schema if x.field_type == "RECORD"]) > 0

# Default arguments are only evaluated once when the function is created
def schema_tracking_factory(singleton=SchemaTrackingContextManager()) -> SchemaTrackingContextManager:
    return singleton

nested_data_context_manager: SchemaTrackingContextManager = schema_tracking_factory()

class CMNnested(ABCMeta):
    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, *args, **kwargs):
        if nested_data_context_manager.active():
            #TODO (abeschorner): handle multiple parents
            current_block = args[0].block           
            nested_data_context_manager.block_start = current_block
            res = self.func(*args, **kwargs)
            nested_data_context_manager.block_end = res.block
            parent: nodes.BigFrameNode = current_block        
            nested_data_context_manager.step(parent) # or current_block.expr.node? nodes or blocks?
            nested_data_context_manager.reset_block_markers()
        return res

    @abstractmethod
    def _nesting(self) -> dict:
        ...

    @property
    def nesting(self) -> dict:
        return self._nesting()


def cm_nested(func: Callable):
    def wrapper(*args, **kwargs):
        if nested_data_context_manager.active():
            #TODO (abeschorner): handle multiple parents
            current_block = args[0].block           
            nested_data_context_manager.block_start = current_block
            prev_op = nested_data_context_manager.prev_changes()
            prev_num_ops = nested_data_context_manager.num_nested_commands
            res = func(*args, **kwargs)
            if (prev_op == nested_data_context_manager.prev_changes()) | (nested_data_context_manager.num_nested_commands <= prev_num_ops):
                raise ValueError(f"Nested operation {func.__qualname__} did not set changes in the nested_data_context_manager!")
                
            nested_data_context_manager.block_end = res.block
            parent: nodes.BigFrameNode = current_block        
            nested_data_context_manager.step() # or current_block.expr.node? nodes or blocks?
            nested_data_context_manager.reset_block_markers()
        return res
    return wrapper


@dataclass(frozen=True)
class ArrayValue:
    """
    ArrayValue is an immutable type representing a 2D array with per-column types.
    """

    node: nodes.BigFrameNode

    @classmethod
    def from_pyarrow(cls, arrow_table: pa.Table, session: Session):
        adapted_table = local_data.adapt_pa_table(arrow_table)
        schema = local_data.arrow_schema_to_bigframes(adapted_table.schema)

        iobytes = io.BytesIO()
        pa_feather.write_feather(adapted_table, iobytes)
        node = nodes.ReadLocalNode(
            iobytes.getvalue(),
            data_schema=schema,
            session=session,
        )
        if nested_data_context_manager.active():
            nested_data_context_manager.add_source(node)

        return cls(node)

    @classmethod
    def from_cached(
        cls,
        original: ArrayValue,
        table: google.cloud.bigquery.Table,
        ordering: orderings.ExpressionOrdering,
    ):
        node = nodes.CachedTableNode(
            original_node=original.node,
            project_id=table.reference.project,
            dataset_id=table.reference.dataset_id,
            table_id=table.reference.table_id,
            physical_schema=tuple(table.schema),
            ordering=ordering,
        )
        if nested_data_context_manager.active():
            nested_data_context_manager.add_source(node)
        return cls(node)

    @classmethod
    def from_table(
        cls,
        table: google.cloud.bigquery.Table,
        schema: schemata.ArraySchema,
        session: Session,
        *,
        predicate: Optional[str] = None,
        at_time: Optional[datetime.datetime] = None,
        primary_key: Sequence[str] = (),
        offsets_col: Optional[str] = None,
    ):
        if offsets_col and primary_key:
            raise ValueError("must set at most one of 'offests', 'primary_key'")
        if any(i.field_type == "JSON" for i in table.schema if i.name in schema.names):
            warnings.warn(
                "Interpreting JSON column(s) as StringDtype. This behavior may change in future versions.",
                bigframes.exceptions.PreviewWarning,
            )
        node = nodes.ReadTableNode(
            project_id=table.reference.project,
            dataset_id=table.reference.dataset_id,
            table_id=table.reference.table_id,
            physical_schema=tuple(table.schema),
            total_order_cols=(offsets_col,) if offsets_col else tuple(primary_key),
            order_col_is_sequential=(offsets_col is not None),
            columns=schema,
            at_time=at_time,
            table_session=session,
            sql_predicate=predicate,
        )
        if nested_data_context_manager.active():
            nested_data_context_manager.add_source(node)
        return cls(node)

    @property
    def column_ids(self) -> typing.Sequence[str]:
        return self.schema.names

    @property
    def session(self) -> Session:
        required_session = self.node.session
        from bigframes import get_global_session

        return (
            required_session if (required_session is not None) else get_global_session()
        )

    @functools.cached_property
    def schema(self) -> schemata.ArraySchema:
        return self.node.schema

    @functools.cached_property
    def _compiled_schema(self) -> schemata.ArraySchema:
        return bigframes.core.compile.test_only_ibis_inferred_schema(self.node)

    def as_cached(
        self: ArrayValue,
        cache_table: google.cloud.bigquery.Table,
        ordering: Optional[orderings.ExpressionOrdering],
    ) -> ArrayValue:
        """
        Replace the node with an equivalent one that references a tabel where the value has been materialized to.
        """
        node = nodes.CachedTableNode(
            original_node=self.node,
            project_id=cache_table.reference.project,
            dataset_id=cache_table.reference.dataset_id,
            table_id=cache_table.reference.table_id,
            physical_schema=tuple(cache_table.schema),
            ordering=ordering,
        )
        return ArrayValue(node)

    def _try_evaluate_local(self):
        """Use only for unit testing paths - not fully featured. Will throw exception if fails."""
        return bigframes.core.compile.test_only_try_evaluate(self.node)

    def get_column_type(self, key: str) -> bigframes.dtypes.Dtype:
        return self.schema.get_type(key)

    def row_count(self) -> ArrayValue:
        """Get number of rows in ArrayValue as a single-entry ArrayValue."""
        return ArrayValue(nodes.RowCountNode(child=self.node))

    # Operations
    def filter_by_id(self, predicate_id: str, keep_null: bool = False) -> ArrayValue:
        """Filter the table on a given expression, the predicate must be a boolean series aligned with the table expression."""
        predicate: ex.Expression = ex.free_var(predicate_id)
        if keep_null:
            predicate = ops.fillna_op.as_expr(predicate, ex.const(True))
        return self.filter(predicate)

    def filter(self, predicate: ex.Expression):
        return ArrayValue(nodes.FilterNode(child=self.node, predicate=predicate))

    def order_by(self, by: Sequence[OrderingExpression]) -> ArrayValue:
        return ArrayValue(nodes.OrderByNode(child=self.node, by=tuple(by)))

    def reversed(self) -> ArrayValue:
        return ArrayValue(nodes.ReversedNode(child=self.node))

    def promote_offsets(self, col_id: str) -> ArrayValue:
        """
        Convenience function to promote copy of column offsets to a value column. Can be used to reset index.
        """
        if not self.session._strictly_ordered:
            raise ValueError("Generating offsets not supported in unordered mode")
        return ArrayValue(nodes.PromoteOffsetsNode(child=self.node, col_id=col_id))

    def concat(self, other: typing.Sequence[ArrayValue]) -> ArrayValue:
        """Append together multiple ArrayValue objects."""
        return ArrayValue(
            nodes.ConcatNode(children=tuple([self.node, *[val.node for val in other]]))
        )

    def project_to_id(self, expression: ex.Expression, output_id: str):
        if output_id in self.column_ids:  # Mutate case
            exprs = [
                ((expression if (col_id == output_id) else ex.free_var(col_id)), col_id)
                for col_id in self.column_ids
            ]
        else:  # append case
            self_projection = (
                (ex.free_var(col_id), col_id) for col_id in self.column_ids
            )
            exprs = [*self_projection, (expression, output_id)]
        return ArrayValue(
            nodes.ProjectionNode(
                child=self.node,
                assignments=tuple(exprs),
            )
        )

    def assign(self, source_id: str, destination_id: str) -> ArrayValue:
        if destination_id in self.column_ids:  # Mutate case
            exprs = [
                (
                    (
                        ex.free_var(source_id)
                        if (col_id == destination_id)
                        else ex.free_var(col_id)
                    ),
                    col_id,
                )
                for col_id in self.column_ids
            ]
        else:  # append case
            self_projection = (
                (ex.free_var(col_id), col_id) for col_id in self.column_ids
            )
            exprs = [*self_projection, (ex.free_var(source_id), destination_id)]
        return ArrayValue(
            nodes.ProjectionNode(
                child=self.node,
                assignments=tuple(exprs),
            )
        )

    def assign_constant(
        self,
        destination_id: str,
        value: typing.Any,
        dtype: typing.Optional[bigframes.dtypes.Dtype],
    ) -> ArrayValue:
        if pandas.isna(value):
            # Need to assign a data type when value is NaN.
            dtype = dtype or bigframes.dtypes.DEFAULT_DTYPE

        if destination_id in self.column_ids:  # Mutate case
            exprs = [
                (
                    (
                        ex.const(value, dtype)
                        if (col_id == destination_id)
                        else ex.free_var(col_id)
                    ),
                    col_id,
                )
                for col_id in self.column_ids
            ]
        else:  # append case
            self_projection = (
                (ex.free_var(col_id), col_id) for col_id in self.column_ids
            )
            exprs = [*self_projection, (ex.const(value, dtype), destination_id)]
        return ArrayValue(
            nodes.ProjectionNode(
                child=self.node,
                assignments=tuple(exprs),
            )
        )

    def select_columns(self, column_ids: typing.Sequence[str]) -> ArrayValue:
        selections = ((ex.free_var(col_id), col_id) for col_id in column_ids)
        return ArrayValue(
            nodes.ProjectionNode(
                child=self.node,
                assignments=tuple(selections),
            )
        )

    def drop_columns(self, columns: Iterable[str]) -> ArrayValue:
        new_projection = (
            (ex.free_var(col_id), col_id)
            for col_id in self.column_ids
            if col_id not in columns
        )
        return ArrayValue(
            nodes.ProjectionNode(
                child=self.node,
                assignments=tuple(new_projection),
            )
        )

    def aggregate(
        self,
        aggregations: typing.Sequence[typing.Tuple[ex.Aggregation, str]],
        by_column_ids: typing.Sequence[str] = (),
        dropna: bool = True,
    ) -> ArrayValue:
        """
        Apply aggregations to the expression.
        Arguments:
            aggregations: input_column_id, operation, output_column_id tuples
            by_column_id: column id of the aggregation key, this is preserved through the transform
            dropna: whether null keys should be dropped
        """
        return ArrayValue(
            nodes.AggregateNode(
                child=self.node,
                aggregations=tuple(aggregations),
                by_column_ids=tuple(by_column_ids),
                dropna=dropna,
            )
        )

    def project_window_op(
        self,
        column_name: str,
        op: agg_ops.UnaryWindowOp,
        window_spec: WindowSpec,
        output_name=None,
        *,
        never_skip_nulls=False,
        skip_reproject_unsafe: bool = False,
    ) -> ArrayValue:
        """
        Creates a new expression based on this expression with unary operation applied to one column.
        column_name: the id of the input column present in the expression
        op: the windowable operator to apply to the input column
        window_spec: a specification of the window over which to apply the operator
        output_name: the id to assign to the output of the operator, by default will replace input col if distinct output id not provided
        never_skip_nulls: will disable null skipping for operators that would otherwise do so
        skip_reproject_unsafe: skips the reprojection step, can be used when performing many non-dependent window operations, user responsible for not nesting window expressions, or using outputs as join, filter or aggregation keys before a reprojection
        """
        # TODO: Support non-deterministic windowing
        if window_spec.row_bounded or not op.order_independent:
            if not self.session._strictly_ordered:
                raise ValueError(
                    "Order-dependent windowed ops not supported in unordered mode"
                )
        return ArrayValue(
            nodes.WindowOpNode(
                child=self.node,
                column_name=column_name,
                op=op,
                window_spec=window_spec,
                output_name=output_name,
                never_skip_nulls=never_skip_nulls,
                skip_reproject_unsafe=skip_reproject_unsafe,
            )
        )

    def _reproject_to_table(self) -> ArrayValue:
        """
        Internal operators that projects the internal representation into a
        new ibis table expression where each value column is a direct
        reference to a column in that table expression. Needed after
        some operations such as window operations that cannot be used
        recursively in projections.
        """
        return ArrayValue(
            nodes.ReprojectOpNode(
                child=self.node,
            )
        )

    def unpivot(
        self,
        row_labels: typing.Sequence[typing.Hashable],
        unpivot_columns: typing.Sequence[
            typing.Tuple[str, typing.Tuple[typing.Optional[str], ...]]
        ],
        *,
        passthrough_columns: typing.Sequence[str] = (),
        index_col_ids: typing.Sequence[str] = ["index"],
        join_side: typing.Literal["left", "right"] = "left",
    ) -> ArrayValue:
        """
        Unpivot ArrayValue columns.

        Args:
            row_labels: Identifies the source of the row. Must be equal to length to source column list in unpivot_columns argument.
            unpivot_columns: Mapping of column id to list of input column ids. Lists of input columns may use None.
            passthrough_columns: Columns that will not be unpivoted. Column id will be preserved.
            index_col_id (str): The column id to be used for the row labels.

        Returns:
            ArrayValue: The unpivoted ArrayValue
        """
        # There will be N labels, used to disambiguate which of N source columns produced each output row
        explode_offsets_id = bigframes.core.guid.generate_guid("unpivot_offsets_")
        labels_array = self._create_unpivot_labels_array(
            row_labels, index_col_ids, explode_offsets_id
        )

        # Unpivot creates N output rows for each input row, labels disambiguate these N rows
        joined_array = self._cross_join_w_labels(labels_array, join_side)

        # Build the output rows as a case statment that selects between the N input columns
        unpivot_exprs = []
        # Supports producing multiple stacked ouput columns for stacking only part of hierarchical index
        for col_id, input_ids in unpivot_columns:
            # row explode offset used to choose the input column
            # we use offset instead of label as labels are not necessarily unique
            cases = itertools.chain(
                *(
                    (
                        ops.eq_op.as_expr(explode_offsets_id, ex.const(i)),
                        ex.free_var(id_or_null)
                        if (id_or_null is not None)
                        else ex.const(None),
                    )
                    for i, id_or_null in enumerate(input_ids)
                )
            )
            col_expr = ops.case_when_op.as_expr(*cases)
            unpivot_exprs.append((col_expr, col_id))

        label_exprs = ((ex.free_var(id), id) for id in index_col_ids)
        # passthrough columns are unchanged, just repeated N times each
        passthrough_exprs = ((ex.free_var(id), id) for id in passthrough_columns)
        return ArrayValue(
            nodes.ProjectionNode(
                child=joined_array.node,
                assignments=(*label_exprs, *unpivot_exprs, *passthrough_exprs),
            )
        )

    def _cross_join_w_labels(
        self, labels_array: ArrayValue, join_side: typing.Literal["left", "right"]
    ) -> ArrayValue:
        """
        Convert each row in self to N rows, one for each label in labels array.
        """
        table_join_side = (
            join_def.JoinSide.LEFT if join_side == "left" else join_def.JoinSide.RIGHT
        )
        labels_join_side = table_join_side.inverse()
        labels_mappings = tuple(
            join_def.JoinColumnMapping(labels_join_side, id, id)
            for id in labels_array.schema.names
        )
        table_mappings = tuple(
            join_def.JoinColumnMapping(table_join_side, id, id)
            for id in self.schema.names
        )
        join = join_def.JoinDefinition(
            conditions=(), mappings=(*labels_mappings, *table_mappings), type="cross"
        )
        if join_side == "left":
            joined_array = self.join(labels_array, join_def=join)
        else:
            joined_array = labels_array.join(self, join_def=join)
        return joined_array

    def _create_unpivot_labels_array(
        self,
        former_column_labels: typing.Sequence[typing.Hashable],
        col_ids: typing.Sequence[str],
        offsets_id: str,
    ) -> ArrayValue:
        """Create an ArrayValue from a list of label tuples."""
        rows = []
        for row_offset, row_label in enumerate(former_column_labels):
            row_label = (row_label,) if not isinstance(row_label, tuple) else row_label
            row = {
                col_ids[i]: (row_label[i] if pandas.notnull(row_label[i]) else None)
                for i in range(len(col_ids))
            }
            row[offsets_id] = row_offset
            rows.append(row)

        return ArrayValue.from_pyarrow(pa.Table.from_pylist(rows), session=self.session)

    def join(
        self,
        other: ArrayValue,
        join_def: join_def.JoinDefinition,
        allow_row_identity_join: bool = False,
    ):
        join_node = nodes.JoinNode(
            left_child=self.node,
            right_child=other.node,
            join=join_def,
            allow_row_identity_join=allow_row_identity_join,
        )
        if allow_row_identity_join:
            return ArrayValue(bigframes.core.rewrite.maybe_rewrite_join(join_node))
        return ArrayValue(join_node)

    def try_align_as_projection(
        self,
        other: ArrayValue,
        join_type: join_def.JoinType,
        mappings: typing.Tuple[join_def.JoinColumnMapping, ...],
    ) -> typing.Optional[ArrayValue]:
        result = bigframes.core.rewrite.join_as_projection(
            self.node, other.node, mappings, join_type
        )
        if result is not None:
            return ArrayValue(result)
        return None

    def explode(self, column_ids: typing.Sequence[str]) -> ArrayValue:
        assert len(column_ids) > 0
        for column_id in column_ids:
            assert bigframes.dtypes.is_array_like(self.get_column_type(column_id))

        return ArrayValue(
            nodes.ExplodeNode(child=self.node, column_ids=tuple(column_ids))
        )

    def _uniform_sampling(self, fraction: float) -> ArrayValue:
        """Sampling the table on given fraction.

        .. warning::
            The row numbers of result is non-deterministic, avoid to use.
        """
        return ArrayValue(nodes.RandomSampleNode(self.node, fraction))

