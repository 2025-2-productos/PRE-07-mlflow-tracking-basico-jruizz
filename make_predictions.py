#### MAKE_PREDICTIONS ####

"""Prediccion script for the MLflow model.

This script loads a model from MLflow and makes predictions on a dataset.

$ python3 make_predictions.py


"""

import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"


df = pd.read_csv(FILE_PATH)
y = df["quality"]
x = df.drop(columns=["quality"])

## Debe verificarse el run_id del modelo que se quiere cargar
## Se puede obtener el run_id desde la interfaz de MLflow

logged_model = "runs:/bac58733a82b4486a2f68fb0df606e0d/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
y = loaded_model.predict(x)

print(y)
