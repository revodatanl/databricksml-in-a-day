# Databricks notebook source
# MAGIC %md ## IRIS dataset lab
# MAGIC
# MAGIC In this self-guided lab exercise, you will be working with the well-known [iris](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) dataset. We make sure we import every package in the beginning of the notebook. It also gives you an idea of what we will be using. 

# COMMAND ----------

import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# COMMAND ----------

# TODO set the place of the mlflow registry to UC.


# COMMAND ----------

# MAGIC %md ## Ingesting the raw data
# MAGIC
# MAGIC - Using Pandas, read the dataset from `/Volumes/databricks_training/raw/iris/iris.csv` to a variable called `iris_df`.
# MAGIC - Print the first couple of lines to get an idea of the data.
# MAGIC - Make sure you have a good feeling of the dataset and the structure.

# COMMAND ----------



# COMMAND ----------

# TODO: Make sure you convert the `species` column to a `category` type column. Hinst: use `astype`.


# TODO: Create a target variable from the `species` column and assign it to `y`.
# TODO: Using the feature variables, create a new dataframe called `X`.
X = 
y = 

# COMMAND ----------

# TODO: Perform a train-test split. Assign 50% of the data to training. 
# TODO: Make sure your results are reproducible by using a random state


# COMMAND ----------

# MAGIC %md ## Model 1: Decision Tree model
# MAGIC - Create an mlflow experiment. `run_name` should be `untuned_decision_tree`.
# MAGIC - First train a simple `DecisionTreeClassifier` model 
# MAGIC   - using `max_depth=4`.
# MAGIC   - using a random state
# MAGIC - Make sure you log all these with mlflow.
# MAGIC - Once you've trained the model, make sure you log it too. Use the `infer_signature` function and log a signature alongside the model instance.
# MAGIC - Use the model to make predictions on `X_test`.
# MAGIC - Having the predictions, calculate the accuracy score and log it with mlflow.
# MAGIC

# COMMAND ----------



  # Hyperparameters and log params
 

  # Model training and log model
  

  # Making predictions and log metrics


# COMMAND ----------

# MAGIC %md ## Register a model in UC

# COMMAND ----------

# TODO: get the run_id for the model you just logged. 
run_id = 

# COMMAND ----------

catalog_name = "databricks_training"
schema_name = "models"
model_name = # TODO pattern: <yourname>_iris

# TODO: user the run_id to register the model in UC as version 1.
model_version = 

# COMMAND ----------

# MAGIC %md You should now see the model in the UC page. Next, either using the UI, or `MlflowClient`:
# MAGIC - Add the following tag to the model: `{"task": "classification"}`
# MAGIC - Add the following tag to the model version 1: `{"validation_status": "approved"}`
# MAGIC - Add a the following alias to the model version 1: `Champion`

# COMMAND ----------

# TODO:

# COMMAND ----------

# MAGIC %md ## Model 2: Random Forest model
# MAGIC - Create an mlflow experiment. `run_name` should be `untuned_random_forest`.
# MAGIC - Next, train a simple `RandomForestClassifier` model 
# MAGIC   - using `max_depth=10`.
# MAGIC   - using `n_estimators=50`
# MAGIC   - using a random state
# MAGIC - Make sure you log all these with mlflow.
# MAGIC - Once you've trained the model, make sure you log it too. Use the `infer_signature` function and log a signature alongside the model instance.
# MAGIC - Use the model to make predictions on `X_test`.
# MAGIC - Having the predictions, calculate the accuracy score and log it with mlflow.
# MAGIC

# COMMAND ----------

  # Hyperparameters and log params

  # Model training and log model

  # Making predictions and log metrics

# COMMAND ----------

# MAGIC %md ## Transition new model version

# COMMAND ----------

# TODO: get the run_id for the model you just logged. 
run_id = 

# COMMAND ----------

# TODO: register this new model as version number 2 for the model you just registered in UC.
new_model_version = 

# COMMAND ----------

# MAGIC %md You should now see the new model version in the UC page. Next, either using the UI, or `MlflowClient`:
# MAGIC - Add the following tag to model version 2: `{"validation_status": "approved"}`
# MAGIC - Add a the following alias to the model version 1: `Archived`
# MAGIC - Add a tag the following alias to the model version 1: `Champion`

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Make predictions

# COMMAND ----------

# TODO: use the champion model to make predictions
champion_model = 

print(f'Accuracy: {accuracy_score(y_test, champion_model.predict(X_test))}')
