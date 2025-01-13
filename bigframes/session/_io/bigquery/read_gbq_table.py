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

"""
Private helpers for loading a BigQuery table as a BigQuery DataFrames DataFrame.
"""

from __future__ import annotations

import datetime
import typing
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

import bigframes_vendored.constants as constants
import google.api_core.exceptions
import google.cloud.bigquery as bigquery

import bigframes
import bigframes.clients
import bigframes.core.compile
import bigframes.core.compile.default_ordering
import bigframes.core.sql
import bigframes.dtypes
import bigframes.exceptions as bfe
import bigframes.session._io.bigquery
import bigframes.session.clients
import bigframes.version

# Avoid circular imports.
if typing.TYPE_CHECKING:
    import bigframes.session


def get_table_metadata(
    bqclient: bigquery.Client,
    table_ref: google.cloud.bigquery.table.TableReference,
    bq_time: datetime.datetime,
    *,
    cache: Dict[bigquery.TableReference, Tuple[datetime.datetime, bigquery.Table]],
    use_cache: bool = True,
) -> Tuple[datetime.datetime, google.cloud.bigquery.table.Table]:
    """Get the table metadata, either from cache or via REST API."""

    cached_table = cache.get(table_ref)
    if use_cache and cached_table is not None:
        snapshot_timestamp, _ = cached_table

        # Cache hit could be unexpected. See internal issue 329545805.
        # Raise a warning with more information about how to avoid the
        # problems with the cache.
        msg = (
            f"Reading cached table from {snapshot_timestamp} to avoid "
            "incompatibilies with previous reads of this table. To read "
            "the latest version, set `use_cache=False` or close the "
            "current session with Session.close() or "
            "bigframes.pandas.close_session()."
        )
        # There are many layers before we get to (possibly) the user's code:
        # pandas.read_gbq_table
        # -> with_default_session
        # -> Session.read_gbq_table
        # -> _read_gbq_table
        # -> _get_snapshot_sql_and_primary_key
        # -> get_snapshot_datetime_and_table_metadata
        warnings.warn(msg, stacklevel=7)
        return cached_table

    table = bqclient.get_table(table_ref)
    # local time will lag a little bit do to network latency
    # make sure it is at least table creation time.
    # This is relevant if the table was created immediately before loading it here.
    if (table.created is not None) and (table.created > bq_time):
        bq_time = table.created

    cached_table = (bq_time, table)
    cache[table_ref] = cached_table
    return cached_table


def validate_table(
    bqclient: bigquery.Client,
    table: bigquery.table.Table,
    columns: Optional[Sequence[str]],
    snapshot_time: datetime.datetime,
    filter_str: Optional[str] = None,
) -> bool:
    """Validates that the table can be read, returns True iff snapshot is supported."""

    time_travel_not_found = False
    # Anonymous dataset, does not support snapshot ever
    if table.dataset_id.startswith("_"):
        pass
    # Only true tables support time travel
    elif table.table_type != "TABLE":
        if table.table_type == "MATERIALIZED_VIEW":
            msg = (
                "Materialized views do not support FOR SYSTEM_TIME AS OF queries. "
                "Attempting query without time travel. Be aware that as materialized views "
                "are updated periodically, modifications to the underlying data in the view may "
                "result in errors or unexpected behavior."
            )
            warnings.warn(msg, category=bfe.TimeTravelDisabledWarning)
    else:
        # table might support time travel, lets do a dry-run query with time travel
        snapshot_sql = bigframes.session._io.bigquery.to_query(
            query_or_table=f"{table.reference.project}.{table.reference.dataset_id}.{table.reference.table_id}",
            columns=columns or (),
            sql_predicate=filter_str,
            time_travel_timestamp=snapshot_time,
        )
        try:
            # If this succeeds, we don't need to query without time travel, that would surely succeed
            bqclient.query_and_wait(
                snapshot_sql, job_config=bigquery.QueryJobConfig(dry_run=True)
            )
            return True
        except google.api_core.exceptions.NotFound:
            # note that a notfound caused by a simple typo will be
            # caught above when the metadata is fetched, not here
            time_travel_not_found = True

    # At this point, time travel is known to fail, but can we query without time travel?
    snapshot_sql = bigframes.session._io.bigquery.to_query(
        query_or_table=f"{table.reference.project}.{table.reference.dataset_id}.{table.reference.table_id}",
        columns=columns or (),
        sql_predicate=filter_str,
        time_travel_timestamp=None,
    )
    # Any erorrs here should just be raised to user
    bqclient.query_and_wait(
        snapshot_sql, job_config=bigquery.QueryJobConfig(dry_run=True)
    )
    if time_travel_not_found:
        msg = (
            "NotFound error when reading table with time travel."
            " Attempting query without time travel. Warning: Without"
            " time travel, modifications to the underlying table may"
            " result in errors or unexpected behavior."
        )
        warnings.warn(msg, category=bfe.TimeTravelDisabledWarning)
    return False


def are_index_cols_unique(
    bqclient: bigquery.Client,
    table: bigquery.table.Table,
    index_cols: List[str],
    api_name: str,
    metadata_only: bool = False,
) -> bool:
    if len(index_cols) == 0:
        return False
    # If index_cols contain the primary_keys, the query engine assumes they are
    # provide a unique index.
    primary_keys = frozenset(_get_primary_keys(table))
    if (len(primary_keys) > 0) and primary_keys <= frozenset(index_cols):
        return True

    if metadata_only:
        # Sometimes not worth scanning data to check uniqueness
        return False
    # TODO(b/337925142): Avoid a "SELECT *" subquery here by ensuring
    # table_expression only selects just index_cols.
    is_unique_sql = bigframes.core.sql.is_distinct_sql(index_cols, table.reference)
    job_config = bigquery.QueryJobConfig()
    job_config.labels["bigframes-api"] = api_name
    results = bqclient.query_and_wait(is_unique_sql, job_config=job_config)
    row = next(iter(results))

    return row["total_count"] == row["distinct_count"]


def _get_primary_keys(
    table: bigquery.table.Table,
) -> List[str]:
    """Get primary keys from table if they are set."""

    primary_keys: List[str] = []
    if (
        (table_constraints := getattr(table, "table_constraints", None)) is not None
        and (primary_key := table_constraints.primary_key) is not None
        # This will be False for either None or empty list.
        # We want primary_keys = None if no primary keys are set.
        and (columns := primary_key.columns)
    ):
        primary_keys = columns if columns is not None else []

    return primary_keys


def _is_table_clustered_or_partitioned(
    table: bigquery.table.Table,
) -> bool:
    """Returns True if the table is clustered or partitioned."""

    # Could be None or an empty tuple if it's not clustered, both of which are
    # falsey.
    if table.clustering_fields:
        return True

    if (
        time_partitioning := table.time_partitioning
    ) is not None and time_partitioning.type_ is not None:
        return True

    if (
        range_partitioning := table.range_partitioning
    ) is not None and range_partitioning.field is not None:
        return True

    return False


def get_index_cols(
    table: bigquery.table.Table,
    index_col: Iterable[str] | str | bigframes.enums.DefaultIndexKind,
) -> List[str]:
    """
    If we can get a total ordering from the table, such as via primary key
    column(s), then return those too so that ordering generation can be
    avoided.
    """

    # Transform index_col -> index_cols so we have a variable that is
    # always a list of column names (possibly empty).
    if isinstance(index_col, bigframes.enums.DefaultIndexKind):
        if index_col == bigframes.enums.DefaultIndexKind.SEQUENTIAL_INT64:
            # User has explicity asked for a default, sequential index.
            # Use that, even if there are primary keys on the table.
            return []
        if index_col == bigframes.enums.DefaultIndexKind.NULL:
            return []
        else:
            # Note: It's actually quite difficult to mock this out to unit
            # test, as it's not possible to subclass enums in Python. See:
            # https://stackoverflow.com/a/33680021/101923
            raise NotImplementedError(
                f"Got unexpected index_col {repr(index_col)}. {constants.FEEDBACK_LINK}"
            )
    elif isinstance(index_col, str):
        index_cols: List[str] = [index_col]
    else:
        index_cols = list(index_col)

    # If the isn't an index selected, use the primary keys of the table as the
    # index. If there are no primary keys, we'll return an empty list.
    if len(index_cols) == 0:
        primary_keys = _get_primary_keys(table)

        # If table has clustering/partitioning, fail if we haven't been able to
        # find index_cols to use. This is to avoid unexpected performance and
        # resource utilization because of the default sequential index. See
        # internal issue 335727141.
        if _is_table_clustered_or_partitioned(table) and not primary_keys:
            msg = (
                f"Table '{str(table.reference)}' is clustered and/or "
                "partitioned, but BigQuery DataFrames was not able to find a "
                "suitable index. To avoid this warning, set at least one of: "
                # TODO(b/338037499): Allow max_results to override this too,
                # once we make it more efficient.
                "`index_col` or `filters`."
            )
            warnings.warn(msg, category=bfe.DefaultIndexWarning)

        # If there are primary keys defined, the query engine assumes these
        # columns are unique, even if the constraint is not enforced. We make
        # the same assumption and use these columns as the total ordering keys.
        index_cols = primary_keys

    return index_cols


def get_time_travel_datetime_and_table_metadata(
    bqclient: bigquery.Client,
    table_ref: bigquery.TableReference,
    *,
    api_name: str,
    cache: Dict[bigquery.TableReference, Tuple[datetime.datetime, bigquery.Table]],
    use_cache: bool = True,
) -> Tuple[datetime.datetime, bigquery.Table]:
    cached_table = cache.get(table_ref)
    if use_cache and cached_table is not None:
        snapshot_timestamp, _ = cached_table

        # Cache hit could be unexpected. See internal issue 329545805.
        # Raise a warning with more information about how to avoid the
        # problems with the cache.
        msg = (
            f"Reading cached table from {snapshot_timestamp} to avoid "
            "incompatibilies with previous reads of this table. To read "
            "the latest version, set `use_cache=False` or close the "
            "current session with Session.close() or "
            "bigframes.pandas.close_session()."
        )
        # There are many layers before we get to (possibly) the user's code:
        # pandas.read_gbq_table
        # -> with_default_session
        # -> Session.read_gbq_table
        # -> _read_gbq_table
        # -> _get_snapshot_sql_and_primary_key
        # -> get_snapshot_datetime_and_table_metadata
        warnings.warn(msg, stacklevel=7)
        return cached_table

    # TODO(swast): It's possible that the table metadata is changed between now
    # and when we run the CURRENT_TIMESTAMP() query to see when we can time
    # travel to. Find a way to fetch the table metadata and BQ's current time
    # atomically.
    table = bqclient.get_table(table_ref)

    job_config = bigquery.QueryJobConfig()
    job_config.labels["bigframes-api"] = api_name
    snapshot_timestamp = list(
        bqclient.query(
            "SELECT CURRENT_TIMESTAMP() AS `current_timestamp`",
            job_config=job_config,
        ).result()
    )[0][0]
    cached_table = (snapshot_timestamp, table)
    cache[table_ref] = cached_table
    return cached_table
