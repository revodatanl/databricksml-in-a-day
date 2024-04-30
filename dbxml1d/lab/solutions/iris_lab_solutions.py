# Databricks notebook source
# MAGIC %md ## IRIS dataset lab
# MAGIC
# MAGIC In this self-guided lab exercise, you will be working with the well-known [iris](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) dataset. We make sure we import every package in the beginning of the notebook. It also gives you an idea of what we will be using. 

# COMMAND ----------

import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# COMMAND ----------

# TODO set the place of the mlflow registry to UC.
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md ## Ingesting the raw data
# MAGIC
# MAGIC - Using Pandas, read the dataset from `/Volumes/databricks_training/raw/iris/iris.csv` to a variable called `iris_df`.
# MAGIC - Print the first couple of lines to get an idea of the data.
# MAGIC - Make sure you have a good feeling of the dataset and the structure.

# COMMAND ----------

iris_df = pd.read_csv("/Volumes/databricks_training/raw/iris/iris.csv", sep=",")

iris_df.head()

# COMMAND ----------

# TODO: Make sure you convert the `species` column to a `category` type column. Hinst: use `astype`.
iris_df['species'] = iris_df['species'].astype('category')

# TODO: Create a target variable from the `species` column and assign it to `y`.
# TODO: Using the feature variables, create a new dataframe called `X`.
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_df['species']

# COMMAND ----------

# TODO: Perform a train-test split. Assign 50% of the data to training. 
# TODO: Make sure your results are reproducible by using a random state
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)

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

with mlflow.start_run(run_name='untuned_decision_tree'):

  # Hyperparameters and log params
  max_depth = 4
  mlflow.log_params({"max_depth": max_depth, "model_type": "tree", "random_state": 42})

  # Model training and log model
  dt_model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
  dt_model.fit(X_train, y_train)
  signature = infer_signature(X_train, y_train)
  mlflow.sklearn.log_model(sk_model=dt_model, artifact_path="decision_tree_model", signature=signature)

  # Making predictions and log metrics
  predictions = dt_model.predict(X_test)
  accuracy = accuracy_score(predictions, y_test)
  print(accuracy)
  mlflow.log_metrics({"accuracy": accuracy})

# COMMAND ----------

# MAGIC %md ## Register a model in UC

# COMMAND ----------

# TODO: get the run_id for the model you just logged. 
run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_decision_tree"').iloc[0].run_id
run_id

# COMMAND ----------

catalog_name = "databricks_training"
schema_name = "models"
model_name = "kornelkovacs_iris" # TODO pattern: <yourname>_iris

# TODO: user the run_id to register the model in UC as version 1.
model_version = mlflow.register_model(f"runs:/{run_id}/decision_tree_model", f"{catalog_name}.{schema_name}.{model_name}")


# COMMAND ----------

# MAGIC %md You should now see the model in the UC page. Next, either using the UI, or `MlflowClient`:
# MAGIC - Add the following tag to the model: `{"task": "classification"}`
# MAGIC - Add the following tag to the model version 1: `{"validation_status": "approved"}`
# MAGIC - Add a the following alias to the model version 1: `Champion`

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

client.set_registered_model_tag(f"{catalog_name}.{schema_name}.{model_name}", "task", "classification")

client.set_registered_model_alias(f"{catalog_name}.{schema_name}.{model_name}", "Champion", 1)

client.set_model_version_tag(f"{catalog_name}.{schema_name}.{model_name}", "1", "validation_status", "approved")

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

with mlflow.start_run(run_name='untuned_random_forest'):

  # Hyperparameters and log params
  max_depth = 4
  n_estimators = 100
  mlflow.log_params({"max_depth": max_depth, 
                     "model_type": "ensemble", 
                     "random_state": 42,
                     "n_estimators": n_estimators})

  # Model training and log model
  rf_model = RandomForestClassifier(random_state=42, max_depth=max_depth, n_estimators=n_estimators)
  rf_model.fit(X_train, y_train)
  signature = infer_signature(X_train, y_train)
  mlflow.sklearn.log_model(sk_model=rf_model, artifact_path="random_forest_model", signature=signature)

  # Making predictions and log metrics
  predictions = rf_model.predict(X_test)
  accuracy = accuracy_score(predictions, y_test)
  print(accuracy)
  mlflow.log_metrics({"accuracy": accuracy})

# COMMAND ----------

# MAGIC %md ## Transition new model version

# COMMAND ----------

# TODO: get the run_id for the model you just logged. 
run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id
run_id

# COMMAND ----------

# TODO: register this new model as version number 2 for the model you just registered in UC.
new_model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", f"{catalog_name}.{schema_name}.{model_name}")

# COMMAND ----------

# MAGIC %md You should now see the new model version in the UC page. Next, either using the UI, or `MlflowClient`:
# MAGIC - Add the following tag to model version 2: `{"validation_status": "approved"}`
# MAGIC - Add a the following alias to the model version 1: `Archived`
# MAGIC - Add a tag the following alias to the model version 1: `Champion`

# COMMAND ----------

client.set_model_version_tag(f"{catalog_name}.{schema_name}.{model_name}", "2", "validation_status", "approved")

client.set_registered_model_alias(f"{catalog_name}.{schema_name}.{model_name}", "Champion", 2)

client.set_registered_model_alias(f"{catalog_name}.{schema_name}.{model_name}", "Archived", 1)

# COMMAND ----------

# MAGIC %md ## Make predictions

# COMMAND ----------

# TODO: use the champion model to make predictions
champion_model = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.{model_name}@Champion")

print(f'Accuracy: {accuracy_score(y_test, champion_model.predict(X_test))}')
