from sklearn.datasets import load_iris
import altair as alt
import click
import pandas as pd
import altair as alt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import mlflow.sklearn
import altair as alt
import altair_viewer

# Internal Libraries
import mlflow_vismod




df_iris = load_iris(as_frame=True)

viz_iris = (
    alt.Chart(df_iris)
      .mark_circle(size=60)
      .encode(x="x", y="y", color="z:N")
      .properties(height=375, width=575)
      .interactive()
)

mlflow_vismod.log_model(
    model=viz_iris,
    artifact_path="viz",
    style="vegalite",
    input_example=df_iris.head(5),
)
