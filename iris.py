import mlflow
import argparse
import sys
from mlflow import pyfunc
import pandas as pd
import shutil
import tempfile
import mlflow.tensorflow
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
from xgboost import XGBClassifier 
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import shap
import mlflow
import mlflow.xgboost
import os
import dvc.api


mlflow.set_tracking_uri(os.environ['MLFLOWURI'])
mlflow.set_experiment("cml-dvc experiment")
repo=os.getcwd()
train_resource_url = dvc.api.get_url(
    path='train.csv',
    repo=repo,
  

    )

df = pd.read_csv(train_resource_url)
X =  df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = df[['variety']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "learning_rate": 0.3,
            "eval_metric": "mlogloss",
            "colsample_bytree": 1.0,
            "subsample": 1.0,
            "seed": 42,
        }
with mlflow.start_run():
    mlflow.log_param('data_url',train_resource_url)
    #model = xg.fit(params, dtrain, evals=[(dtrain, "train")])
    model = XGBClassifier(params)
    mlflow.log_params(params)
    model.fit(X_train, y_train)
    mlflow.xgboost.log_model(model,"model")
    # evaluate model
    y_pred = model.predict(X_test)
    #loss = log_loss(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy",acc)

    cm = confusion_matrix(y_test, y_pred, normalize='all',)
    cmd = ConfusionMatrixDisplay(cm,display_labels=["Setosa", "Versicolor", "Virginica"])
    cmd.plot()

    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    plt.clf()
    plt.barh(['sepal length (cm)',
     'sepal width (cm)',
     'petal length (cm)',
     'petal width (cm)']
    , model.feature_importances_)
    plt.savefig('featureimportance.jpg')
    mlflow.log_artifact('featureimportance.jpg')
    plt.clf()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar",show=False)
    plt.savefig("shap.png")
    mlflow.log_artifact('shap.png')
    plt.clf()




