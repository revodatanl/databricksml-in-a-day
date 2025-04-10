# Databricks notebook source
# MAGIC %md # Training machine learning models on tabular data: an end-to-end example
# MAGIC
# MAGIC This tutorial covers the following steps:
# MAGIC - Visualize the data using Seaborn and matplotlib
# MAGIC - Run a parallel hyperparameter sweep to train machine learning models on the dataset
# MAGIC - Explore the results of the hyperparameter sweep with MLflow
# MAGIC - Register the best performing model in MLflow
# MAGIC - Apply the registered model to another dataset using a Spark UDF
# MAGIC - Set up model serving for low-latency requests
# MAGIC
# MAGIC In this example, you build a model to predict the quality of Portugese "Vinho Verde" wine based on the wine's physicochemical properties. 
# MAGIC
# MAGIC The example uses a dataset from the UCI Machine Learning Repository, presented in [*Modeling wine preferences by data mining from physicochemical properties*](https://www.sciencedirect.com/science/article/pii/S0167923609001377?via%3Dihub) [Cortez et al., 2009].
# MAGIC
# MAGIC ## Requirements
# MAGIC This notebook requires Databricks Runtime for Machine Learning. This notebook requires a workspace that has been enabled for Unity Catalog. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog setup

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

CATALOG_NAME = "ml_training"
SCHEMA_NAME = "models"
USERNAME = 'magdalena_brach'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the data

# COMMAND ----------

import pandas as pd

white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

# COMMAND ----------

# MAGIC %md Merge the two DataFrames into a single dataset, with a new binary feature "is_red" that indicates whether the wine is red or white.

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# Remove spaces from column names
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# COMMAND ----------

data.head()

# COMMAND ----------

# MAGIC %md ## Visualize data
# MAGIC
# MAGIC Before training a model, explore the dataset using Seaborn and Matplotlib.

# COMMAND ----------

# MAGIC %md Plot a histogram of the dependent variable, quality.

# COMMAND ----------

import seaborn as sns
sns.distplot(data.quality, kde=False)

# COMMAND ----------

# MAGIC %md Looks like quality scores are normally distributed between 3 and 9. 
# MAGIC
# MAGIC Define a wine as high quality if it has quality >= 7.

# COMMAND ----------

display(data)

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

# MAGIC %md Box plots are useful in noticing correlations between features and a binary label. Create box plots for each feature to compare high-quality and low-quality wines. Significant differences in the box plots indicate good predictors of quality.

# COMMAND ----------

import matplotlib.pyplot as plt

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
  if col == 'is_red' or col == 'quality':
    continue # Box plots cannot be used on indicator variables
  sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0

# COMMAND ----------

# MAGIC %md In the above box plots, a few variables stand out as good univariate predictors of quality. 
# MAGIC
# MAGIC - In the alcohol box plot, the median alcohol content of high quality wines is greater than even the 75th quantile of low quality wines. High alcohol content is correlated with quality.
# MAGIC - In the density box plot, low quality wines have a greater density than high quality wines. Density is inversely correlated with quality.

# COMMAND ----------

# MAGIC %md ## Preprocess data
# MAGIC Prior to training a model, check for missing values and split the data into training and validation sets.

# COMMAND ----------

data.isna().any()

# COMMAND ----------

# MAGIC %md There are no missing values.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare dataset for training baseline model
# MAGIC Split the input data into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = data.drop(["quality"], axis=1)
y = data.quality

# Split out the training data
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)

# Split the remaining data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)

# COMMAND ----------

# MAGIC %md ## Build a baseline model
# MAGIC This task seems well suited to a random forest classifier, since the output is binary and there may be interactions between multiple variables.
# MAGIC
# MAGIC The following code builds a simple classifier using scikit-learn. It uses MLflow to keep track of the model accuracy, and to save the model for later use.

# COMMAND ----------

import cloudpickle
import time

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import mlflow.pyfunc
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env


# The predict method of sklearn's RandomForestClassifier returns a binary classification (0 or 1). 
# The following code creates a wrapper function, SklearnModelWrapper, that uses 
# the predict_proba method to return the probability that the observation belongs to each class. 

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

# mlflow.start_run creates a new MLflow run to track the performance of this model. 
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.
with mlflow.start_run(run_name='untuned_random_forest'):
  n_estimators = 10   #The number of trees in the forest.
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)

  # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  mlflow.log_param('n_estimators', n_estimators)

  # Use the area under the ROC curve as a metric.
  mlflow.log_metric('auc', auc_score)
  wrappedModel = SklearnModelWrapper(model)

  # Log the model with a signature that defines the schema of the model's inputs and outputs. 
  # When the model is deployed, this signature will be used to validate inputs.
  signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
  
  mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, signature=signature)

# COMMAND ----------

# MAGIC %md Examine the learned feature importances output by the model as a sanity-check.

# COMMAND ----------

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md As illustrated by the boxplots shown previously, both alcohol and density are important in predicting quality.

# COMMAND ----------

# MAGIC %md You logged the Area Under the ROC Curve (AUC) to MLflow. Click **Experiment** at the upper right to display the Experiment Runs sidebar. 
# MAGIC
# MAGIC The model achieved an AUC of 0.854.
# MAGIC
# MAGIC A random classifier would have an AUC of 0.5, and higher AUC values are better. For more information, see [Receiver Operating Characteristic Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve).

# COMMAND ----------

# MAGIC %md #### Register the model in MLflow UC Model Registry
# MAGIC
# MAGIC By registering this model in UC Model Registry, you can easily reference the model from anywhere within Databricks.
# MAGIC
# MAGIC The following section shows how to do this programmatically, but you can also register a model using the UI. See "Create or register a model using the UI" ([AWS](https://docs.databricks.com/applications/machine-learning/manage-model-lifecycle/index.html#create-or-register-a-model-using-the-ui)|[Azure](https://docs.microsoft.com/azure/databricks/applications/machine-learning/manage-model-lifecycle/index#create-or-register-a-model-using-the-ui)|[GCP](https://docs.gcp.databricks.com/applications/machine-learning/manage-model-lifecycle/index.html#create-or-register-a-model-using-the-ui)).

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id

# COMMAND ----------

# If you see the error "PERMISSION_DENIED: User does not have any permission level assigned to the registered model", 
# the cause may be that a model already exists with the name "wine_quality". Try using a different name.

model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{USERNAME}_wine_quality"

model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", f"{model_name}")

# Registering the model takes a few seconds, so add a small delay
time.sleep(15)

# COMMAND ----------

# MAGIC %md You should now see the model in the Models page. To display the Models page, click the Models icon in the left sidebar. 
# MAGIC
# MAGIC Next, transition this model to production and load it into this notebook from Model Registry.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

client.set_registered_model_tag(f"{model_name}", "task", "classification")

client.set_registered_model_alias(f"{model_name}", "Champion", 1)

client.set_model_version_tag(f"{model_name}", "1", "validation_status", "approved")

# COMMAND ----------

# MAGIC %md In Unity Catalog, the model version now has the tag "Champion".
# MAGIC
# MAGIC You can now refer to the model using the path "models:/{model_name}@Champion".

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}@Champion")

# Sanity-check: This should match the AUC logged by MLflow
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md ##Experiment with a new model
# MAGIC
# MAGIC The random forest model performed well even without hyperparameter tuning.
# MAGIC Use the xgboost library to train a more accurate model. Run a hyperparameter sweep to train multiple models in parallel, using Hyperopt and SparkTrials. As before, MLflow tracks the performance of each parameter configuration.

# COMMAND ----------

from math import exp

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope

import mlflow.xgboost
import xgboost as xgb

search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': 123, # Set a seed for deterministic training
}

def train_model(params):

  # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
  mlflow.xgboost.autolog()

  with mlflow.start_run(nested=True):
    train = xgb.DMatrix(data=X_train, label=y_train)
    validation = xgb.DMatrix(data=X_val, label=y_val)

    # Pass in the validation set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
    # is no longer improving.
    booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                        evals=[(validation, "validation")], early_stopping_rounds=50)
    
    validation_predictions = booster.predict(validation)
    auc_score = roc_auc_score(y_val, validation_predictions)
    mlflow.log_metric('auc', auc_score)

    signature = infer_signature(X_train, booster.predict(train))
    mlflow.xgboost.log_model(booster, "model", signature=signature)
    
    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}

# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials(parallelism=5)

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .
with mlflow.start_run(run_name='xgboost_models'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=30,
    trials=spark_trials,
  )

# COMMAND ----------

# MAGIC %md  
# MAGIC #### Use MLflow to view the results
# MAGIC Open up the Experiment Runs sidebar to see the MLflow runs. Click on Date next to the down arrow to display a menu, and select 'auc' to display the runs sorted by the auc metric. The highest auc value is 0.90.
# MAGIC
# MAGIC MLflow tracks the parameters and performance metrics of each run. Click the External Link icon <img src="https://docs.databricks.com/_static/images/icons/external-link.png"/> at the top of the Experiment Runs sidebar to navigate to the MLflow Runs Table.

# COMMAND ----------

# MAGIC %md Now investigate how the hyperparameter choice correlates with AUC. Click the "+" icon to expand the parent run, then select all runs except the parent, and click "Compare". Select the Parallel Coordinates Plot.
# MAGIC
# MAGIC The Parallel Coordinates Plot is useful in understanding the impact of parameters on a metric. You can drag the pink slider bar at the upper right corner of the plot to highlight a subset of AUC values and the corresponding parameter values. The plot below highlights the highest AUC values:
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/mlflow/end-to-end-example/parallel-coordinates-plot.png"/>
# MAGIC
# MAGIC Notice that all of the top performing runs have a low value for reg_lambda and learning_rate. 
# MAGIC
# MAGIC You could run another hyperparameter sweep to explore even lower values for these parameters. For simplicity, that step is not included in this example.

# COMMAND ----------

# MAGIC %md 
# MAGIC You used MLflow to log the model produced by each hyperparameter configuration. The following code finds the best performing run and saves the model to Model Registry.
# MAGIC

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
print(f'AUC of Best Run: {best_run["metrics.auc"]}')

# COMMAND ----------

# MAGIC %md #### Update the production `wine_quality` model in MLflow Model Registry
# MAGIC
# MAGIC Earlier, you saved the baseline model to Model Registry with the name `wine_quality`. Now that you have a created a more accurate model, update `wine_quality`.

# COMMAND ----------

new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", f"{model_name}")

# Registering the model takes a few seconds, so add a small delay
time.sleep(15)

# COMMAND ----------

# MAGIC %md Click **Models** in the left sidebar to see that the `wine_quality` model now has two versions. 
# MAGIC
# MAGIC The following code promotes the new version to production.

# COMMAND ----------

client.set_registered_model_alias(f"{model_name}", "champion", 2)

# COMMAND ----------

client.set_registered_model_alias(f"{model_name}", "archived", 1)

# COMMAND ----------

# MAGIC %md Clients that call load_model now receive the new model.

# COMMAND ----------

# This code is the same as the last block of "Building a Baseline Model". No change is required for clients to get the new model!
model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md The auc value on the test set for the new model is 0.90. You beat the baseline!

# COMMAND ----------

# MAGIC %md ##Batch inference
# MAGIC
# MAGIC There are many scenarios where you might want to evaluate a model on a corpus of new data. For example, you may have a fresh batch of data, or may need to compare the performance of two models on the same corpus of data.
# MAGIC
# MAGIC The following code evaluates the model on data stored in a Delta table, using Spark to run the computation in parallel.

# COMMAND ----------

# To simulate a new corpus of data, save the existing X_train data to a Delta table. 
# In the real world, this would be a new batch of data.
spark_df = spark.createDataFrame(X_train)

table_name = f"{CATALOG_NAME}.batch_data.{USERNAME}_wine_data"

(spark_df
  .write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema",True)
  .saveAsTable(table_name)
)

# COMMAND ----------

# MAGIC %md Load the model into a Spark UDF, so it can be applied to the Delta table.

# COMMAND ----------

import mlflow.pyfunc

apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}@champion")

# COMMAND ----------

# Read the "new data" from the Unity Catalog table
new_data = spark.read.table(f"{CATALOG_NAME}.batch_data.{USERNAME}_wine_data")

# COMMAND ----------

display(new_data)

# COMMAND ----------

from pyspark.sql.functions import struct

# Apply the model to the new data
udf_inputs = struct(*(X_train.columns.tolist()))

new_data = new_data.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)

# COMMAND ----------

# Each row now has an associated prediction. Note that the xgboost function does not output probabilities by default, so the predictions are not limited to the range [0, 1].
display(new_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve the model
# MAGIC To productionize the model for low latency predictions, use Mosaic AI Model Serving to deploy the model to an endpoint. The following cell shows how to use the MLflow Deployments SDK to create a model serving endpoint

# COMMAND ----------

# from mlflow.deployments import get_deploy_client

# client = get_deploy_client("databricks")
# endpoint = client.create_endpoint(
#     name="wine-model-endpoint",
#     config={
#         "served_entities": [
#             {
#                 "name": "wine-entity",
#                 "entity_name": model_name,
#                 "entity_version": "1",
#                 "workload_size": "Small",
#                 "scale_to_zero_enabled": True
#             }
#         ],
#       }
# )

# COMMAND ----------

# MAGIC %md ## Cleanup

# COMMAND ----------

versions=[1, 2]
for version in versions:
  client.delete_model_version(name=f"{model_name}", version=version)

client.delete_registered_model(name=f"{model_name}")