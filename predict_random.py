import mlflow
logged_model = 'runs:/70ed96c6005845ce80adee025bdf5622/iris_rf'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = pd.read_csv("./test_iris.csv")
loaded_model.predict(data)