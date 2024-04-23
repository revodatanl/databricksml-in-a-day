# Databricks notebook source
# MAGIC %md # Basic example for Feature Engineering in Unity Catalog
# MAGIC This notebook illustrates how you can use Databricks Feature Engineering in Unity Catalog to create, store, and manage Unity Catalog Features to train ML models and make batch predictions, including with features whose value is only available at the time of prediction. In this example, the goal is to predict the wine quality using a ML model with a variety of static wine features and a realtime input.
# MAGIC
# MAGIC This notebook shows how to:
# MAGIC - Create a feature table and use it to build a training dataset for a machine learning model.
# MAGIC - Modify the feature table and use the updated table to create a new version of the model.
# MAGIC - Use the Databricks Features UI to determine how features relate to models.
# MAGIC - Perform batch scoring using automatic feature lookup.
# MAGIC
# MAGIC ## Requirements
# MAGIC - Databricks Runtime 13.2 for Machine Learning or above.
# MAGIC   - If you do not have access to Databricks Runtime for Machine Learning, you can run this notebook on Databricks Runtime 13.2 or above. To do so, run `%pip install databricks-feature-engineering` at the start of this notebook.
# MAGIC

# COMMAND ----------

import pandas as pd

from pyspark.sql.functions import monotonically_increasing_id, expr, rand
import uuid

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# COMMAND ----------

# MAGIC %md ## Load dataset
# MAGIC The code in the following cell loads the dataset and does some minor data preparation: creates a unique ID for each observation and removes spaces from the column names. The unique ID column (`wine_id`) is the primary key of the feature table and is used to lookup features.

# COMMAND ----------

raw_data = spark.read.load("/databricks-datasets/wine-quality/winequality-red.csv",format="csv",sep=";",inferSchema="true",header="true" )

def addIdColumn(dataframe, id_column_name):
    """Add id column to dataframe"""
    columns = dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df

def renameColumns(df):
    """Rename columns to be compatible with Feature Engineering in UC"""
    renamed_df = df
    for column in df.columns:
        renamed_df = renamed_df.withColumnRenamed(column, column.replace(' ', '_'))
    return renamed_df

# Run functions
renamed_df = renameColumns(raw_data)
df = addIdColumn(renamed_df, 'wine_id')

# Drop target column ('quality') as it is not included in the feature table
features_df = df.drop('quality')
display(features_df)


# COMMAND ----------

# MAGIC %md ## Create a new catalog or reuse an existing catalog
# MAGIC To create a new catalog, you must have the `CREATE CATALOG` privilege on the metastore.
# MAGIC To use an existing catalog, you must have the `USE CATALOG` privilege on the catalog.

# COMMAND ----------

# Create a new catalog with:
catalog_name = "ml_training"
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"USE CATALOG {catalog_name}")

# COMMAND ----------

# MAGIC %md ## Create a new schema in the catalog
# MAGIC To create a new schema in the catalog, you must have the `CREATE SCHEMA` privilege on the catalog.

# COMMAND ----------

schema_name = "featurestore_schema"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
spark.sql(f"USE SCHEMA {schema_name}")

# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"{catalog_name}.{schema_name}.wine_db_" + str(uuid.uuid4())[:6]
print(table_name)

# COMMAND ----------

# MAGIC %md ## Create the feature table

# COMMAND ----------

# MAGIC %md The first step is to create a FeatureEngineeringClient.

# COMMAND ----------

fe = FeatureEngineeringClient()

# You can get help in the notebook for feature engineering client API functions:
# help(fe.<function_name>)

# For example:
# help(fe.create_table)

# COMMAND ----------

# MAGIC %md Create the feature table. For a complete API reference, see ([AWS](https://docs.databricks.com/machine-learning/feature-store/python-api.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/python-api)|[GCP](https://docs.gcp.databricks.com/machine-learning/feature-store/python-api.html)).

# COMMAND ----------

fe.create_table(
    name=table_name,
    primary_keys=["wine_id"],
    df=features_df,
    schema=features_df.schema,
    description="wine features"
)

# COMMAND ----------

# MAGIC %md You can also use `create_table` without providing a dataframe, and then later populate the feature table using `fe.write_table`.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC ```
# MAGIC fe.create_table(
# MAGIC     name=table_name,
# MAGIC     primary_keys=["wine_id"],
# MAGIC     schema=features_df.schema,
# MAGIC     description="wine features"
# MAGIC )
# MAGIC
# MAGIC fe.write_table(
# MAGIC     name=table_name,
# MAGIC     df=features_df,
# MAGIC     mode="merge"
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## Train a model with Feature Engineering in Unity Catalog

# COMMAND ----------

# MAGIC %md The feature table does not include the prediction target. However, the training dataset needs the prediction target values. There may also be features that are not available until the time the model is used for inference.
# MAGIC
# MAGIC This example uses the feature **`real_time_measurement`** to represent a characteristic of the wine that can only be observed at inference time. This feature is used in training and the feature value for a wine is provided at inference time.

# COMMAND ----------

## inference_data_df includes wine_id (primary key), quality (prediction target), and a real time feature
inference_data_df = df.select("wine_id", "quality", (10 * rand()).alias("real_time_measurement"))
display(inference_data_df)

# COMMAND ----------

# MAGIC %md Use a `FeatureLookup` to build a training dataset that uses the specified `lookup_key` to lookup features from the feature table and the online feature `real_time_measurement`. If you do not specify the `feature_names` parameter, all features except the primary key are returned.

# COMMAND ----------

def load_data(table_name, lookup_key):
    # In the FeatureLookup, if you do not provide the `feature_names` parameter, all features except primary keys are returned
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fe.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
    training_set = fe.create_training_set(df=inference_data_df, feature_lookups=model_feature_lookups, label="quality", exclude_columns="wine_id")
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("quality", axis=1)
    y = training_pd["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

# Create the train and test datasets
X_train, X_test, y_train, y_test, training_set = load_data(table_name, "wine_id")
X_train.head()

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

# Configure MLflow client to access models in Unity Catalog
mlflow.set_registry_uri("databricks-uc")

model_name = f"{catalog_name}.models.wine_model"

client = MlflowClient()

try:
    client.delete_registered_model(model_name) # Delete the model if already created
except:
    None

# COMMAND ----------

# MAGIC %md
# MAGIC The code in the next cell trains a scikit-learn RandomForestRegressor model and logs the model with the Feature Engineering in UC.
# MAGIC
# MAGIC The code starts an MLflow experiment to track training parameters and results. Note that model autologging is disabled (`mlflow.sklearn.autolog(log_models=False)`); this is because the model is logged using `fe.log_model`.

# COMMAND ----------

# Disable MLflow autologging and instead log the model using Feature Engineering in UC
mlflow.sklearn.autolog(log_models=False)

def train_model(X_train, X_test, y_train, y_test, training_set, fe):
    ## fit and log model
    with mlflow.start_run() as run:

        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))

        fe.log_model(
            model=rf,
            artifact_path="wine_quality_prediction",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=model_name,
        )

train_model(X_train, X_test, y_train, y_test, training_set, fe)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC To view the logged model, navigate to the MLflow Experiments page for this notebook. To access the Experiments page, click the Experiments icon on the left navigation bar:  <img src="https://docs.databricks.com/_static/images/icons/experiments-icon.png"/>
# MAGIC
# MAGIC Find the notebook experiment in the list. It has the same name as the notebook, in this case, "Basic example for Feature Engineering in Unity Catalog".
# MAGIC
# MAGIC Click the experiment name to display the experiment page. The packaged Feature Engineering in UC model, created when you called `fe.log_model` appears in the **Artifacts** section of this page. You can use this model for batch scoring.
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/basic-fs-nb-artifact.png"/>
# MAGIC
# MAGIC The model is also automatically registered in the Unity Catalog.

# COMMAND ----------

# MAGIC %md ## Batch scoring
# MAGIC Use `score_batch` to apply a packaged Feature Engineering in UC model to new data for inference. The input data only needs the primary key column `wine_id` and the realtime feature `real_time_measurement`. The model automatically looks up all of the other feature values from the feature tables.

# COMMAND ----------

display(inference_data_df.drop("quality"))

# COMMAND ----------


