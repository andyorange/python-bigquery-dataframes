from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping
from collections import deque
from typing import final, TYPE_CHECKING, Tuple
from typing_extensions import Self   #TODO (abeschorner): deprecated, please move away from python 3.9. Add Self import to typing
from networkx import DiGraph, topological_sort, compose as nx_compose
from pyarrow import DataType as pa_datatype

# -- big frames and local imports --
from google.cloud.bigquery.schema import SchemaField
from bigframes._config import options as config_options
import bigframes_vendored.pandas.pandas._typing as vendored_pandas_typing
from bigframes.dataframe import DataFrame, Literal, Sequence
import bigframes.core.blocks as blocks
from bigframes.dtypes import is_struct_like
import bigframes.session

if TYPE_CHECKING:
    from bigframes.dataframe import DataFrame

class NestedDataError(Exception):
    def __init__(self, message: str, 
                 params: list|None=None, error_exc: str|None=None):
        txt = " | ".join(params) if params is not None else None
        msg = f"{message}: [{txt}]" if txt is not None else f"{message}"
        msg = f"Nested data error -- {msg}"
        if error_exc:
            msg += f"\n Raised from: {error_exc}"
        #self.message = msg
        super().__init__(msg)


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
    def __init__(self, sep_layers: str | None = None, sep_structs: str | None = None):
        # TODO: change into parameters
        # this needs to be done before getting the schema
        assert(config_options.bigquery.project is not None and config_options.bigquery.location is not None)
        self.sep_layers = sep_layers if sep_layers is not None else SchemaTracker._default_sep_layers
        self.sep_structs = sep_structs if sep_structs is not None else SchemaTracker._default_sep_structs
        self.schema_orig: Tuple[SchemaField, ...] | None = None
        self.schema = {}
        self.dag: DiGraph = DiGraph()
        self.root = None

    @staticmethod
    def leaves(dag: DiGraph):
        return [node for node in dag.nodes if dag.out_degree(node) == 0]

    @staticmethod
    def has_nested_data(schema: Tuple[SchemaField, ...]) -> bool:
        return sum([1 for x in schema if x.field_type == "RECORD"]) > 0

    @staticmethod
    def _tree_from_strings(paths: list[str], struct_separator: str) -> dict:
        root = {}
        for path in paths:
            parts = path.split(struct_separator)
            node = root
            for part in parts:
                node = node.setdefault(part, {})
        return root

    @property
    def is_valid(self) -> bool:
        """
        Returns True if self._dag is not None, which is the case whenever the ArrayValue's BigFrameNode has a physical_schema attribute.
        Other cases will be handled in the near future.
        """
        return self.dag is not None

    @property
    def schema_lineage(self):
        return list(self.dag)

    def _init_dag_from_df_schema(self, dag: DiGraph, schema: dict[str, tuple], layer_separator: str, struct_separator: str) -> DiGraph:
        dag_ret = dag
        root = [el for el in topological_sort(dag)][0]

        for key, value in schema.items():
            parent = value[0] if value[0] else root
            dag_ret.add_node(key, node_type=schema[key])
            dag_ret.add_edge(parent, key)

        #TODO: Debug log info
        #print([el for el in topological_sort(dag)])
        return dag_ret    

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
                _parent = [p for p in prefixes if pref == p.rsplit(self.sep_layers, 1)[0]]
                _parent = _parent[0].rstrip(self.sep_layers) if len(_parent) > 0 else parent
                if bigframes.dtypes.is_struct_like(dtp):
                    nested_col.append(col)
                    prefixes.append(col+self.sep_layers)
                value = tuple((_parent, dtp))
                if schema_ret.get(col, None) is None:
                    schema_ret[col] = value
                if nested_col:
                    continue  # restart after having exploded
            if nested_col:
                df_flattened = df_flattened.struct.explode(nested_col[0], separator=self.sep_layers)
        # finalize adding non nested columns to schema
        for col, dtp in schema.items():
            if schema_ret.get(col, None) is None:
                schema_ret[col] = tuple(("", dtp))
            
        return tuple((df_flattened, schema_ret)) # type: ignore

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

    ## -- public methods / API --

    def dag_from_df(self, schema: dict[str, tuple],
                   struct_separator: str, layer_separator: str) -> DiGraph:
        dag = DiGraph()
        dag.add_node(self._base_root_name, node_type=self._base_root_name)
        #dag_dict = self._tree_from_strings(cols_flattened, struct_separator=layer_separator)
        dag_res = self._init_dag_from_df_schema(dag, schema, layer_separator=layer_separator, struct_separator=struct_separator)
        return dag_res

    def to_name_layer(self, col: str) -> str:
        return col.replace(self.sep_structs, self.sep_layers)

    def to_name_struct(self, col: str) -> str:
        return col.replace(self.sep_layers, self.sep_structs)

    def merge(self, lineage: SchemaTracker) -> DiGraph:
        "merges 'other' lineage into self by adding all its root-successors to self.root"
        assert( (self.sep_layers == lineage.sep_layers) and (self.sep_structs == lineage.sep_structs))
        dag_merged = nx_compose(self.dag, lineage.dag)
        return dag_merged

    def add(self, df: DataFrame, layer_separator: str, struct_separator: str):
        df_flattened, df_schema = self._explode_nested(df, sep_explode=layer_separator, sep_struct=struct_separator)
        schema = df_flattened.dtypes
        cols = list(df_flattened.dtypes.keys())
        node = df_flattened._block.expr.node
        dag = self.dag_from_df(schema=df_schema, layer_separator=layer_separator, struct_separator=struct_separator)
        schema_orig: Tuple[SchemaField, ...] = node.physical_schema if hasattr(node, "physical_schema") else None # type: ignore, TODO: drop physical schema for now!
        source = SchemaSource(node_flattened=node, dag=dag, schema=schema, schema_orig=schema_orig) # type: ignore  # noqa: E999
        hash_df = bfnode_hash(df._block.expr.node)
        self._source_handler.add_source(hash_df=hash_df, source = source)
        return

    def start_lineage(self, df: DataFrame):
        df_flattened, df_schema = self._explode_nested(df)
        self.schemata.append(df_schema)
        # add root note to initialize DAG
        self.dag.add_node(self._base_root_name, node_type=self._base_root_name)
        self.root = [el for el in topological_sort(self.dag)][0]
        for key, value in df_schema.items():
            parent = value[0] if value[0] else self.root
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

    def extend_dag(self, changes: dict|Mapping):
        dag_leaves = set(SchemaTracker.leaves(dag=self.dag)) # type: ignore
        source_leaves = set([self.to_name_layer(col) for col in list(changes.keys())])
        if not source_leaves.issubset(dag_leaves):
            invalid_cols = [self.to_name_layer(col) for col in source_leaves if col not in dag_leaves]
            raise NestedDataError("Invalid columns/ Columns not in schema", params=invalid_cols)
        _changes = SchemaTracker._parse_change_levels(changes, self.sep_structs)
        _changes = {self.to_name_layer(_from): self.to_name_layer(_to) for _from, _to in _changes.items()}
        for col_source, col_target in changes.items():
            self.dag.add_node(col_target)
            self.dag.add_edge(col_source, col_target)
        # different implementation
        source = self._source_handler.get(hdl_child)
        if source is None:
            self._add_child(hdl_parent, hdl_child)
            source = self._source_handler.get(hdl_child)
            assert(source is not None)
            #raise NestedDataError("Unknown nested node handel", params=[hdl.__qualname__])
        dag_leaves = set(self._source_handler.leaves(dag=source.dag))
        source_leaves = set([self._to_name_layer(col) for col in list(changes.keys())])
        if not source_leaves.issubset(dag_leaves):
            invalid_cols = [self._to_name_layer(col) for col in source_leaves if col not in dag_leaves]
            raise NestedDataError("Invalid columns/ Columns not in schema", params=invalid_cols)
        _changes = SchemaTrackingContextManager._parse_change_levels(changes, self.sep_structs)
        _changes = {self._to_name_layer(_from): self._to_name_layer(_to) for _from, _to in _changes.items()}
        source.extend_dag(_changes)

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
        if not self.sep_structs in level:  # noqa: E713, we are looking for character(s) within a string
            return [level]
        queue = deque([(list(self.dag.nodes)[0], 0, list(self.dag.nodes)[0])])
        leaves = []
        target_level_nodes = []
        while queue:
            node, node_level, node_path = queue.popleft()
            if node_path == level:
                target_level_nodes.append(node)
            elif node in target_level_nodes:
                if not self.dag[node]:  # node is a leaf
                    leaves.append(node)
            else:
                for child in self.dag.nodes(node, []):
                    queue.append((child, node_level + 1, node_path + "__" + child))
        return leaves

    @property
    def num_nested_commands(self) -> int:
        return self._op_count

    def prev_changes(self) -> tuple[nodes.BigFrameNode, dict|Mapping]:
        return ((self._calling_node, self._latest_op)) # type: ignore

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
    # @staticmethod
    # def nesting_op(func: Callable):
    #     def wrapper(*args, **kwargs):
    #         print(f"Calling {func.__name__}")
    #         self.lineage.add_changes(NestedDataFrame.rename.__qualname__, columns, fct=DataFrame.rename)
    #         result = func(*args, **kwargs)
    #         print(f"Finished {func.__name__}")
    #         return result
    #     return wrapper
                   
    def _init_(self, data=None, index: vendored_pandas_typing.Axes | None = None,
        columns: vendored_pandas_typing.Axes | None = None, dtype: bigframes.dtypes.DtypeString | bigframes.dtypes.Dtype | None = None,
        copy: bool | None= None, session: bigframes.session.Session | None = None):

        DataFrame.__init__(self, index=index, columns=columns, dtype=dtype, copy=copy, session=session)
        self.lineage = SchemaTracker()

    def _unroll_columns(self, cols: Sequence[blocks.Label|str] | blocks.Label | str) -> list[str]:
        cols = [str(cols)] if not isinstance(cols, Iterable) else [str(col) for col in cols]
        ret = []
        for col in cols:
            ret += self.lineage.get_leaves_at_level(col)
        return ret

    def _matching_columns(self, left: blocks.Label | Sequence[blocks.Label|str], right: blocks.Label | Sequence[blocks.Label|str]) -> tuple[Sequence[blocks.Label|str], Sequence[blocks.Label|str]]:
        """
        Returns list of "matching" columns from left and right. Matching in this context refers to matching left and right
        lists of columns for length. For nested data, the final level/layer is taken. In other words, all final leaves
        of such a level will be gathered, and as for ordinary pandas, the i-th column of left will be related to the i-th of right.
        Nodes accessible via multiple paths for individual entries (DAG after merging for instance) will only be considered once, as otherwise
        things would become faulty.
        """
        leaves_left = self._unroll_columns(left) # type: ignore
        leaves_right = self._unroll_columns(right) # type: ignore
        assert(len(leaves_left) == len(leaves_right))
        return (leaves_left, leaves_right)

    def rename(self, *, columns: Mapping[blocks.Label, blocks.Label]) -> DataFrame:
        """
        Rename is special in the context of nested data, as we allow column name changes for struct columns!
        Thus we cannot just call it but have to forward it to the context manager via the add_changes method
        """
        self.lineage.add_changes(NestedDataFrame.rename.__qualname__, columns, fct=DataFrame.rename)
        return super().rename(columns=columns)

    def merge(self, right: "NestedDataFrame|DataFrame",
                how: Literal["inner", "left", "outer", "right", "cross"] = "inner", *,
            # TODO(garrettwu): Currently can take inner, outer, left and right. To support
            # cross joins
            on: blocks.Label | str | Sequence[blocks.Label|str] | None = None,
            left_on: blocks.Label | str | Sequence[blocks.Label|str] | None = None,
            right_on: blocks.Label | str | Sequence[blocks.Label|str] | None = None,
            sort: bool = False,
            suffixes: tuple[str, str] = ("_x", "_y")
        ) -> Self:

        _on = self._unroll_columns(on)
        _left_on = self._unroll_columns(left_on)
        _right_on = self._unroll_columns(right_on)
        # ensure all leaves are valid columns
        assert(set(_left_on).issubset(self.columns))
        assert(set(_right_on).issubset(right.columns))
        assert(set(_on).issubset(self.columns) and set(_on).issubset(right.columns))
        df_ret = super().merge(right=right, on=_on, left_on=_left_on, right_on=_right_on, how=how, sort=sort, suffixes=suffixes)
        
        #df_unrolled = super().merge(rirhg, )
        
        return self

# to guarantee singleton creation, we prohibit inheritnig from the class
#nested_data_context_manager: SchemaTrackingContextManager = schema_tracking_factory()
