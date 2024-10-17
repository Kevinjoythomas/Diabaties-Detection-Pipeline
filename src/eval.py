import yaml
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse
import mlflow

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/kevinjoythomas2004/DiabitiesPrediction.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "kevinjoythomas2004"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "8bd5e66c1f291325fc581a214b0bbc85caacfbd0"

params = yaml.safe_load(open("params.yaml"))["train"]


def eval(data_path,model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=['Outcome'])
    y=data['Outcome']
    
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    
    model = pickle.load(open(model_path,'rb'))
    
    pred = model.predict(X)
    acucuracy = accuracy_score(pred,y)
    
    mlflow.log_metric("ACCURACY",acucuracy)
    
if __name__ == "__main__":
    eval(params['data'],params['model'])