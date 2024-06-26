# Copyright 2024 Google LLC
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

import dataclasses
import typing
from typing import TYPE_CHECKING

import bigframes.core.compile.googlesql.abc as abc
import bigframes.core.compile.googlesql.expression as expr

if TYPE_CHECKING:
    import google.cloud.bigquery as bigquery

"""This module provides a structured representation of GoogleSQL syntax using nodes.
Each node's name and child nodes are designed to strictly follow the official GoogleSQL
syntax rules outlined in the documentation:
https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax"""


@dataclasses.dataclass
class QueryExpr(abc.SQLSyntax):
    """This class represents GoogleSQL `query_expr` syntax."""

    select: Select
    with_cte_list: typing.Sequence[NonRecursiveCTE] = ()

    def sql(self) -> str:
        text = []
        if len(self.with_cte_list) > 0:
            with_cte_text = ",\n".join(
                [with_cte.sql() for with_cte in self.with_cte_list]
            )
            text.append(f"WITH {with_cte_text}")

        text.append(self.select.sql())
        return "\n".join(text)


@dataclasses.dataclass
class Select(abc.SQLSyntax):
    """This class represents GoogleSQL `select` syntax."""

    select_list: typing.Sequence[typing.Union[SelectExpression, SelectAll]]
    from_clause_list: typing.Sequence[FromClause] = ()
    distinct: bool = False

    def sql(self) -> str:
        text = ["SELECT"]

        if self.distinct:
            text.append("DISTINCT")

        select_list_sql = ",\n".join([select.sql() for select in self.select_list])
        text.append(select_list_sql)

        if self.from_clause_list is not None:
            from_clauses_sql = ",\n".join(
                [clause.sql() for clause in self.from_clause_list]
            )
            text.append(f"FROM\n{from_clauses_sql}")
        return "\n".join(text)


@dataclasses.dataclass
class SelectExpression(abc.SQLSyntax):
    """This class represents `select_expression`."""

    expression: expr.ColumnExpression
    alias: typing.Optional[expr.AliasExpression] = None

    def sql(self) -> str:
        if self.alias is None:
            return self.expression.sql()
        else:
            return f"{self.expression.sql()} AS {self.alias.sql()}"


@dataclasses.dataclass
class SelectAll(abc.SQLSyntax):
    """This class represents `select_all` (aka. `SELECT *`)."""

    expression: expr.StarExpression

    def sql(self) -> str:
        return self.expression.sql()


@dataclasses.dataclass
class FromClause(abc.SQLSyntax):
    """This class represents GoogleSQL `from_clause` syntax."""

    from_item: FromItem

    def sql(self) -> str:
        return self.from_item.sql()


@dataclasses.dataclass
class FromItem(abc.SQLSyntax):
    """This class represents GoogleSQL `from_item` syntax."""

    # Note: Temporarily introduces the `str` type to interact with pre-existing,
    # compiled SQL strings.
    expression: typing.Union[expr.TableExpression, QueryExpr, str, expr.CTEExpression]
    as_alias: typing.Optional[AsAlias] = None

    @classmethod
    def from_table_ref(
        cls,
        table_ref: bigquery.TableReference,
        as_alias: typing.Optional[AsAlias] = None,
    ):
        return cls(
            expression=expr.TableExpression(
                table_id=table_ref.table_id,
                dataset_id=table_ref.dataset_id,
                project_id=table_ref.project,
            ),
            as_alias=as_alias,
        )

    def sql(self) -> str:
        if isinstance(self.expression, (expr.TableExpression, expr.CTEExpression)):
            text = self.expression.sql()
        elif isinstance(self.expression, str):
            text = f"({self.expression})"
        elif isinstance(self.expression, QueryExpr):
            text = f"({self.expression.sql()})"
        else:
            raise ValueError(
                f"Unsupported expression type {type(self.expression).__name__};"
                "expected one of TableExpression, QueryExpr, str, or CTEExpression."
            )

        if self.as_alias is None:
            return text
        else:
            return f"{text} {self.as_alias.sql()}"


@dataclasses.dataclass
class NonRecursiveCTE(abc.SQLSyntax):
    """This class represents GoogleSQL `non_recursive_cte` syntax."""

    cte_name: expr.CTEExpression
    query_expr: QueryExpr

    def sql(self) -> str:
        return f"{self.cte_name.sql()} AS (\n{self.query_expr.sql()}\n)"


@dataclasses.dataclass
class AsAlias(abc.SQLSyntax):
    """This class represents GoogleSQL `as_alias` syntax."""

    alias: expr.AliasExpression

    def sql(self) -> str:
        return f"AS {self.alias.sql()}"
