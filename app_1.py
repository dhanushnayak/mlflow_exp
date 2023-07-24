import os
from random import random,randint
from mlflow import log_artifacts,log_param,log_metric
import mlflow
#mlflow.set_tracking_uri("http://127.0.0.1:4040")
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
mlflow.set_experiment("Direct_sklearn")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
print(mlflow.get_tracking_uri())