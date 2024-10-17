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


def hyper_tuning(X_train,y_train,param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=True)
    grid_search.fit(X_train,y_train)
    return grid_search

params = yaml.safe_load(open("params.yaml"))["train"]

def train(data_path,model_path,random_state,n_estimators,max_depth):
    data = pd.read_csv(data_path)
    X = data.drop("Outcome",axis=1)
    y = data['Outcome']
    
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    
    with mlflow.start_run():
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        sig = infer_signature(X_train,y_train)        
        
        params_grid= {
            'n_estimators':[100,200],
            'max_depth':[5,10,None],
            'min_samples_split':[2,5],
            'min_samples_leaf':[1,2]
        }
        print("SIZE",X_train.shape)
        print("SIZE",y_train.shape)
        
        grid_search = hyper_tuning(X_train,y_train,param_grid=params_grid)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test,y_pred)
        print("ACCURACY",accuracy)
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_param("n_estimators",grid_search.best_params_['n_estimators'])
        mlflow.log_param("max_depth",grid_search.best_params_['max_depth'])
        mlflow.log_param("min_samples_split",grid_search.best_params_['min_samples_split'])
        mlflow.log_param("min_samples_leaf",grid_search.best_params_['min_samples_leaf'])
        
        cm = confusion_matrix(y_test,y_pred)
        cr = classification_report(y_pred,y_test)
        
        mlflow.log_text(str(cm),"confusion matrix.txt")
        mlflow.log_text(str(cr),"classification_report.txt")
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        if tracking_url_type_store!="file":
            mlflow.sklearn.log_model(best_model,"model",signature=sig,registered_model_name="BEST MODEL")
        else:
            mlflow.sklearn.log_model(best_model,"model",signature=sig)

        os.makedirs(os.path.dirname(model_path),exist_ok=True)
        
        filename=model_path
        pickle.dump(best_model,open(filename,'wb'))
        
        print("MODEL SAVED TO",model_path)
        
if __name__ == "__main__" :
    train(params['data'],params['model'],params['random_state'],params['n_estimators'],params['max_depth'])