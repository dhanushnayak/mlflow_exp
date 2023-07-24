import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import mlflow.sklearn
from mlflow.models.signature import infer_signature
mlflow.set_experiment("Direct_sklearn")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
iris = datasets.load_iris()
iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
import sys
iris_train['label']=iris.target

X=iris_train.iloc[:,:-1]
y=iris_train.iloc[:,-1]

x_train,x_test,y_train,y_test= train_test_split(X,y,random_state=34,test_size=0.3)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=n,max_depth=7, random_state=0,)
    clf.fit(x_train,y_train)
    pred =  clf.predict(x_test)
    print("Model with n = {}".format(n))
    auc =  accuracy_score(y_test,pred)
    print("Acc = {}".format(auc))
    mlflow.log_param("n",n)
    mlflow.log_metric("acc",auc)
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":

            mlflow.sklearn.log_model(clf, "model")
    