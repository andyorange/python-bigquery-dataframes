from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from collections import deque
from typing import final
from networkx import DiGraph, topological_sort
from pyarrow import DataType as pa_datatype
from dataclasses import dataclass
from queue import Queue

# -- big frames and local imports --
from bigframes._config import options as config_options
import bigframes_vendored.pandas.pandas._typing as vendored_pandas_typing
from bigframes.dataframe import DataFrame, Literal, Sequence
import bigframes.core.blocks as blocks
import bigframes.dtypes
import bigframes.session

from bigframes.core.nested_context import NestedDataError
#from bigframes.core.nested_context import SchemaTrackingContextManager, schema_tracking_factory
#from bigframes.core import nested_data_context_manager

# class CMNnested(ABCMeta):
#     def __init__(self, func: Callable):
#         self.func = func

#     def __call__(self, *args, **kwargs):
#         if nested_data_context_manager.active():
#             #TODO (abeschorner): handle multiple parents
#             current_block = args[0].block
#             nested_data_context_manager.block_start = current_block
#             ncm_ref = nested_data_context_manager._source_handler.sources.get(current_block, None)
#             if ncm_ref is not None:
#                 res = self.func(*args, **kwargs)
#                 nested_data_context_manager._block_end = res.block
#                 nested_data_context_manager.step() #current_block, res.block, nested_data_context_manager.latest_changes()) # or current_block.expr.node? nodes or blocks?
#                 nested_data_context_manager.reset_block_markers()
#             else:
#                 #TODO: add node to ncm
#                 raise KeyError(f"Unknown nested node {args[0]}")
#         return res

#     @abstractmethod
#     def _nesting(self) -> dict:
#         ...

#     @property
#     def nesting(self) -> dict:
#         return self._nesting()


def cm_nested(func: Callable):
    def wrapper(*args, **kwargs):
        if nested_data_context_manager.active():
            #TODO (abeschorner): handle multiple parents
            current_block = args[0]._block           
            nested_data_context_manager._block_start = current_block
            prev_op = nested_data_context_manager.prev_changes()
            prev_num_ops = nested_data_context_manager.num_nested_commands
            res = func(*args, **kwargs)
            if (prev_op == nested_data_context_manager.prev_changes()) | (nested_data_context_manager.num_nested_commands <= prev_num_ops):
                raise ValueError(f"Nested operation {func.__qualname__} did not set changes in the nested_data_context_manager!")
            nested_data_context_manager._block_end = res._block      
            nested_data_context_manager.step() # or current_block.expr.node? nodes or blocks?
            nested_data_context_manager.reset_block_markers()
        return res
    return wrapper





# Example usage:
tree = {
    "A": ["A__B", "A__C"],
    "A__B": ["A__B__D", "A__B__E"],
    "A__C": ["A__C__F"],
    "A__B__D": [],
    "A__B__E": [],
    "A__C__F": []
}
level_str = "A__B"
leaves = get_leaves_at_level(tree, level_str)
print(leaves)  # Output: ["A__B__D", "A__B__E"]


@final
class SchemaTracker:
    """
    Context manager for schema tracking using command pattern.
    Utilizes a DAG for schema lineage and thus can reconstruct each step of schema changes.
    """
    _base_root_name = "_root_"
    _default_sep_layers: str = "__"  # make sure it is not a substring of any column name!
    _default_sep_structs: str = "."  # not ot be modified by user

    # setup, start schema deduction
    #def __init__(self, data: DataFrame | Series | str | None=None, layer_separator: str | None = None):
    def __init__(self, layer_separator: str | None = None, struct_separator: str | None = None):
        # TODO: change into parameters
        # this needs to be done before getting the schema
        assert(config_options.bigquery.project is not None and config_options.bigquery.location is not None)
        self.layer_separator = layer_separator if layer_separator is not None else SchemaTracker._default_sep_layers
        self.struct_separator = struct_separator if struct_separator is not None else SchemaTracker._default_sep_structs
        self.schemata = []
        self.dag: DiGraph = DiGraph()
        self._latest_op: dict|Mapping = {}  # latest schema changes
        self._func: Callable|None = None
        self._op_count = 0

    @staticmethod
    def leaves(dag: DiGraph):
        return [node for node in dag.nodes if dag.out_degree(node) == 0]

    @property
    def schema_lineage(self):
        return list(self.dag)

    def to_name_layer(self, col: str) -> str:
        return col.replace(self.struct_separator, self.layer_separator)

    def to_name_struct(self, col: str) -> str:
        return col.replace(self.layer_separator, self.struct_separator)

    def _explode_nested(self, df: DataFrame, columns: list|None=None) -> tuple[DataFrame, dict[str, pa_datatype]]:
        """
        :param bigframes.dataframe.DataFrame df: DataFrame to explode
        :param str sep_explode: separator used in exploded representation
        :param str sep_struct: separator used in BigQuery for separating structs. Default: "."
        :param list[str] colums: columns to explode, using sep_struct as a separator
        :returns tuple[bigframes.dataframe.DataFrame, dict[str, str]]: Returns exploded data frame
            and its schema, layers separated by sep_explode

        The methods explodes a potentially nested DataFrame in a BFS like manner:
        We traverse all columns and explode whenever we find a nested/struct like one.
        If one is found and exploded, we restart. This way we can explode all layers without having to select sub-frames,
        iow no depth processing is done at all.
        """
        schema_ret = {}
        df_flattened = df.copy()
        prefixes = []

        nested_col = [""]
        while nested_col:
            schema = df_flattened.dtypes.to_dict()
            assert(isinstance(schema, dict))
            nested_col = []
            parent = ""
            for col, dtp in schema.items(): 
                pref = col.rsplit(self.layer_separator, 1)[0]
                _parent = [p for p in prefixes if pref == p.rsplit(self.layer_separator, 1)[0]]
                _parent = _parent[0].rstrip(self.layer_separator) if len(_parent) > 0 else parent
                if bigframes.dtypes.is_struct_like(dtp):
                    nested_col.append(col)
                    prefixes.append(col+self.layer_separator)
                value = tuple((_parent, dtp))
                if schema_ret.get(col, None) is None:
                    schema_ret[col] = value
                if nested_col:
                    continue  # restart after having exploded
            if nested_col:
                df_flattened = df_flattened.struct.explode(nested_col[0], separator=self.layer_separator)
        # finalize adding non nested columns to schema
        for col, dtp in schema.items():
            if schema_ret.get(col, None) is None:
                schema_ret[col] = tuple(("", dtp))
            
        return tuple((df_flattened, schema_ret)) # type: ignore
    
    def start_lineage(self, df: DataFrame):
        df_flattened, df_schema = self._explode_nested(df)
        self.schemata.append(df_schema)
        #schema_flattened = df_flattened.dtypes
        # add root note to initialize DAG
        self.dag.add_node(self._base_root_name, node_type=self._base_root_name)
        # add schema to dag
        root = [el for el in topological_sort(self.dag)][0]
        for key, value in df_schema.items():
            parent = value[0] if value[0] else root
            self.dag.add_node(key, node_type=df_schema[key])
            self.dag.add_edge(parent, key)

    @staticmethod
    def _parse_change(chg_from: str, chg_to: str, sep: str) -> tuple[str, str, bool]:
        _from = chg_from.split(sep)
        _to = chg_to.split(sep)
        if (len(_from) < 1) or (len(_to) < 1):
            raise NestedDataError("Column name missing")
        if len(_from) != len(_to):
            if len(_to) != 1:
                return tuple((chg_from, chg_to, False)) # type: ignore
            _chg_to = chg_from.rsplit(sep, 1)[0] + sep + chg_to
            return tuple((chg_from, _chg_to, True)) # type: ignore
        equal = sum([val_from==_to[i] for i, val_from in enumerate(_from[0:-1])])==(len(_from)-1)
        return tuple((chg_from, chg_to, equal)) # type: ignore

    @staticmethod
    def _parse_change_levels(changes: dict|Mapping, sep: str) -> dict:
        """
        Changes must be within the same level/layer of nesting.
        However, to make things easier for the user, we allow a simplified notation such that the targets/values
        of the 'changes' dict can just reference the last layer of the keys. Example:
            person.address.city: location
        is the same as
            person.address.city: person.address.location
        """
        _valid, _invalid = {}, {}
        for col_from, col_to in changes.items():
            ret = SchemaTracker._parse_change(col_from, col_to, sep)
            el = {ret[0]: ret[1]}
            if ret[-1]:
                _valid.update(el)
            else:
                _invalid.update(el)
        if _invalid:
            raise NestedDataError(f"Invalid column name change operation {ret[0]} -> {ret[1]}")
        return _valid

    def _extend_dag(self, changes: dict|Mapping):
        dag_leaves = set(SchemaTracker.leaves(dag=self.dag)) # type: ignore
        source_leaves = set([self.to_name_layer(col) for col in list(changes.keys())])
        if not source_leaves.issubset(dag_leaves):
            invalid_cols = [self.to_name_layer(col) for col in source_leaves if col not in dag_leaves]
            raise NestedDataError("Invalid columns/ Columns not in schema", params=invalid_cols)
        _changes = SchemaTracker._parse_change_levels(changes, self.struct_separator)
        _changes = {self.to_name_layer(_from): self.to_name_layer(_to) for _from, _to in _changes.items()}
        for col_source, col_target in changes.items():
            self.dag.add_node(col_target)
            self.dag.add_edge(col_source, col_target)

    def get_leaves_at_level(self, level: str) -> list[str]:
        """
        Returns all leaves at a given level in a BFS tree.
        Args:
            tree (dict): A dictionary representing the tree, where each key is a node
                and its value is a list of its children.
            level_str (str): A string representing the level at which to retrieve leaves.
        Returns:
            list: A list of leaves at the given level.
        """
        queue = deque([(list(self.dag.nodes)[0], 0, list(self.dag.nodes)[0])])
        leaves = []
        target_level_nodes = []
        while queue:
            node, node_level, node_path = queue.popleft()
            if node_path == level:
                target_level_nodes.append(node)
            elif node in target_level_nodes:
                if not tree[node]:  # node is a leaf
                    leaves.append(node)
            else:
                for child in tree.get(node, []):
                    queue.append((child, node_level + 1, node_path + "__" + child))
        return leaves

    @property
    def num_nested_commands(self) -> int:
        return self._op_count

    def add_changes(self, caller: str, changes: dict|Mapping, fct: Callable|None=None):
        self._latest_op = changes
        self._func = fct
        self._op_count += 1
        self._extend_dag(changes=changes)

    def latest_changes(self) -> list:
        return [self._calling_node, self._latest_op, self._op_count]

    # Private helper methods for starting schema deduction and DAG creation
    @staticmethod
    def _has_nested_data(schema: list) -> bool:
        return sum([1 for x in schema if x.field_type == "RECORD"]) > 0
        
# Work In Progress

    def step(self):  #nodes.BigFrameNode):
        hdl = None
        if hash_start:
            hdl = self._source_handler.sources.get(hash_parent, None)
            if hdl is None:
                raise ValueError(f"NestedDataCM: Unknown data source {self._block_start}")
            
        # no join, merge etc., no new source/BigFrameNode
        else:
            if not self._schemata_matching(self._block_start, hdl):
                raise Exception("Internal error: Nested Schema mismatch")
                self._extend_dag(hdl, self._block_end.expr.node)



class NestedDataFrame(DataFrame):
    @staticmethod
    def nesting_op(func: Callable):
        def wrapper(*args, **kwargs):
            print(f"Calling {func.__name__}")
            self.lineage.add_changes(NestedDataFrame.rename.__qualname__, columns, fct=DataFrame.rename)
            result = func(*args, **kwargs)
            print(f"Finished {func.__name__}")
            return result
        return wrapper
                   
    def _init_(self, data=None, index: vendored_pandas_typing.Axes | None = None,
        columns: vendored_pandas_typing.Axes | None = None, dtype: bigframes.dtypes.DtypeString | bigframes.dtypes.Dtype | None = None,
        copy: bool | None= None, session: bigframes.session.Session | None = None):

        DataFrame.__init__(self, index=index, columns=columns, dtype=dtype, copy=copy, session=session)
        self.lineage = SchemaTracker()

    def _unroll_column(self, col: Sequence[blocks.Label]) -> list[str]:

    def _matching_columns(self, left: blocks.Label | Sequence[blocks.Label], right: blocks.Label | Sequence[blocks.Label]) -> Sequence[blocks.Label]:
        """
        Returns list of "matching" columns from left and right. Matching in this context refers to matching left and right
        lists of columns for length. For nested data, the final level/layer is taken. In other words, all final leaves
        of such a level will be gathered.
        Nodes accessible via multiple paths for individual entries (DAG after merging for instance) will only be considered once, as otherwise
        things would become faulty.
        """
        _left = Sequence[left] if isinstance(left, blocks.Label) else left
        _right = Sequence[right] if isinstance(right, blocks.Label) else right
        leaves_left = []
        for 
        self.lineage.get_leaves_at_level()

    def rename(self, *, columns: Mapping[blocks.Label, blocks.Label]) -> DataFrame:
        """
        Rename is special in the context of nested data, as we allow column name changes for struct columns!
        Thus we cannot just call it but have to forward it to the context manager via the add_changes method
        """
        self.lineage.add_changes(NestedDataFrame.rename.__qualname__, columns, fct=DataFrame.rename)
        return super().rename(columns=columns)

    def merge(self, right: DataFrame,
                how: Literal["inner", "left", "outer", "right", "cross"] = "inner", *,
            # TODO(garrettwu): Currently can take inner, outer, left and right. To support
            # cross joins
            on: blocks.Label | Sequence[blocks.Label] | None = None,
            left_on: blocks.Label | Sequence[blocks.Label] | None = None,
            right_on: blocks.Label | Sequence[blocks.Label] | None = None,
            sort: bool = False,
            suffixes: tuple[str, str] = ("_x", "_y")
        ) -> NestedDataFrame:

# to guarantee singleton creation, we prohibit inheritnig from the class
#nested_data_context_manager: SchemaTrackingContextManager = schema_tracking_factory()
