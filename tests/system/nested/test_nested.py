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

#from google.cloud import bigquery
#from bigframes.functions.nested import BQSchemaLayout, SchemaField

#from google.cloud.bigquery_storage_v1 import types as gtypes
#import pytest
#from typing import List

import bigframes.pandas as bfpd
from bigframes.dataframe import DataFrame #nested_data_context_manager
from bigframes.nesteddataframe import NestedDataFrame, set_project

# def table_schema(table_name_full: str) -> List[SchemaField]:
#     project = table_name_full.split(".")[0]
#     client = bigquery.Client(project=project, location="EU")
#     query_job = client.get_table(table_name_full)
#     return query_job.schema


# def test_unroll_schema():  #table_name_full: pytest.CaptureFixture[str]
#     schema = table_schema("gmbigframes.nested.tiny") # "vf-de-aib-prd-cmr-chn-lab.staging.scs_mini")
#     bqs = BQSchemaLayout(schema)
#     bqs.determine_layout() # TODO: add prefix get_ or determine_
#     return bqs
#     #assert isinstance(schema, List[SchemaField])

# def test_nested_cm():
#     bfpd.options.bigquery.project = "gmbigframes"
#     bfpd.options.bigquery.location = "EU"


# def fct_cm(cm: NestedDataFrame):
#     cm._current_data = bfpd.read_gbq(f"SELECT * FROM {table}"),
#     testdf.apply(cm._current_data),
#     bfpd.get_dummies(cm._current_data)   

def create_simple_nested(create: bool) -> DataFrame:
    import pyarrow as pa
    import pandas as pd
    countries = bfpd.Series(["cn", "es", "us"])
    s = bfpd.Series([
            {"version": 1, "project": "pandas"},
            {"version": 2, "project": "pandas"},
            {"version": 1, "project": "numpy"},
        ], dtype=pd.ArrowDtype( pa.struct([("version", pa.int64()), ("project", pa.string())] ))
    )

    downloads = bfpd.Series([100, 200, 300])
    dfp = bfpd.DataFrame({"country": countries, "file": s, "download_count": downloads}, index=None)

    if create:
        dfp.to_gbq("andreas_beschorner.nested_dbg")
    return dfp

def create_complex_nested(create: bool) -> DataFrame:
    import pyarrow as pa
    import pandas as pd

    nested_struct_schema = pa.struct([
        pa.field("city", pa.string()),
        pa.field("country", pa.string())
    ])
    complex_struct_schema = pa.struct([
        pa.field("name", pa.string()),
        pa.field("age", pa.int64()),
        pa.field("address", nested_struct_schema)
    ])
    complex_data = [
        {"name": "Alice", "age": 30, "address": {"city": "New York", "country": "USA"}},
        {"name": "Bob", "age": 25, "address": {"city": "London", "country": "UK"}}
    ]
    complex_df = bfpd.DataFrame({
        "id": pd.Series([1, 2]),
        "person": pd.Series(
            complex_data,
            dtype=pd.ArrowDtype(complex_struct_schema),
        ),
    }, index=[0, 1])

    if create:
        complex_df.to_gbq("andreas_beschorner.nested_complex")
    return complex_df

def create_complex_nested2(create: bool) -> DataFrame:
    import pyarrow as pa
    import pandas as pd

    nested_struct_schema = pa.struct([
        pa.field("city", pa.string()),
        pa.field("country", pa.string())
    ])
    complex_struct_schema = pa.struct([
        pa.field("name", pa.string()),
        pa.field("age", pa.int64()),
        pa.field("address", nested_struct_schema)
    ])
    complex_data = [
        {"name": "Marlice", "age": 30, "address": {"city": "Bentona", "country": "ES"}},
        {"name": "Huego", "age": 47, "address": {"city": "Berlin", "country": "DE"}}
    ]
    complex_df = bfpd.DataFrame({
        "id": pd.Series([1, 2]),
        "person": pd.Series(
            complex_data,
            dtype=pd.ArrowDtype(complex_struct_schema),
        ),
    }, index=[0, 1])

    if create:
        complex_df.to_gbq("andreas_beschorner.nested_complex")
    return complex_df

if __name__ == "__main__":
    #TODO: autodetect if bfpd is already setup and copy proj/loc if availabe
    set_project(project="vf-de-ca-lab", location="europe-west3")
    table="andreas_beschorner.nested_dbg"
    table = "andreas_beschorner.nested_complex" # table="andreas_beschorner.nested_tiny"
    df1 = create_simple_nested(False)
    df2 = create_complex_nested2(False)
    dfn1 = NestedDataFrame(data=df1)
    dfn2 = NestedDataFrame(data=df2)
    dfn_merged = dfn1.merge(right=dfn2, on=["age"], how="inner")
    print(dfn_merged.head())

    # with nested_data_context_manager as ncm:
    #     #df = bfpd.read_gbq(f"SELECT * FROM {table} limit 10")
    #     #ncm.add_df(df)
    #     #df = bfpd.DataFrame(dfp)
    #     df = create_complex_nested(False)

    #     ncm.add(df, layer_separator=ncm.sep_layers, struct_separator=ncm.sep_structs)
    #     print(ncm.lineage(df))
    #     #df_n.to_gbq("andreas_beschorner.nested_complex_half")
    #     try:
    #         df = df.rename(columns={"person.address.county": "person.address.location"})
    #     except NestedDataError as ne:
    #         print(ne)
    #     df = df.rename(columns={"person.address.country": "location"}) #df = df.rename(columns={"person.address.country": "person.address.location"})
    #     lin = ncm.schema_lineage(df)
    #     if lin is not None:
    #         print(lin)
    #     pass
    pass
    