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


def test_linear_regression(random_model_id: str) -> None:
    your_model_id = random_model_id
    # [START bigquery_dataframes_bqml_linear_regression]
    from bigframes.ml.linear_model import LinearRegression
    import bigframes.pandas as bpd

    # Load data from BigQuery
    bq_df = bpd.read_gbq("bigquery-public-data.ml_datasets.penguins")

    # Drop rows with nulls to get training data
    training_data = bq_df.dropna(subset=["body_mass_g"])

    # Specify your feature (or input) columns and the label (or output) column:
    feature_columns = training_data.drop(columns=["body_mass_g"])
    label_columns = training_data[["body_mass_g"]]

    # Create the linear model
    model = LinearRegression()
    model.fit(feature_columns, label_columns)
    model.to_gbq(
        your_model_id,  # For example: "bqml_tutorial.penguins_model"
        replace=True,
    )
    # [END bigquery_dataframes_bqml_linear_regression]
    # [START bigquery_dataframes_bqml_linear_evaluate]
    import bigframes.pandas as bpd

    # Select the model you will be evaluating. `read_gbq_model` loads model data from
    # BigQuery, but you could also use the `model` object from the previous steps.
    model = bpd.read_gbq_model(
        your_model_id,  # For example: "bqml_tutorial.penguins_model"
    )

    # Score the model with input data defined in an earlier step to compare
    # model predictions on feature_columns to true labels in label_columns.
    score = model.score(feature_columns, label_columns)
    # Expected output results:
    # index  mean_absolute_error  mean_squared_error  mean_squared_log_error  median_absolute_error  r2_score  explained_variance
    #   0        227.012237         81838.159892            0.00507                173.080816        0.872377    0.872377
    #   1 rows x columns
    # [END bigquery_dataframes_bqml_linear_evaluate]
    assert feature_columns is not None
    assert label_columns is not None
    assert model is not None
    assert score is not None
