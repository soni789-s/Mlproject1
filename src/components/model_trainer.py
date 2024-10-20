import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
           logging.info("Split training and test input data")
           x_train,y_train,x_test,y_test = train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
           models = {
               "Random Forest":RandomForestRegressor(),
               "Decision Tree":DecisionTreeRegressor(),
               "Gradient Boosting":GradientBoostingRegressor(),
               "Linear Regression":LinearRegression(),
               "XGRegressor":XGBRegressor(),
               "CatBoosting Regressor":CatBoostRegressor(verbose=False),
               "Adaboost Regressor":AdaBoostRegressor()
           }

           params = {
               "Decision Tree" : {
                   "criterion" : ["squared_error","friedman_mse","absolute_error","poisson"]
               },
               "Random Forest" : {
                   "n_estimators" : [8,16,32,64,128,256]
               },
               "Gradient Boosting" : {
                   "learning_rate" : [0.1,0.01,0.05,0.001],
                   "subsample" : [0.6,0.7,0.75,0.8,0.85,0.9]
               },
               "Linear Regression" : {},
                "XGRegressor" : {
                    "learning_rate":[0.1,0.01,0.5,0.001],
                    "n_estimators" : [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    "depth" :[8,8,10],
                    "learning_rate":[0.01,0.05,0.1],
                    "iterations":[30,50,100]
                },
                "Adaboost Regressor":{
                    "learning_rate":[0.01,0.5,0.1,0.001],
                    "n_estimators" : [8,16,32,64,128,256]
                }
           }
           model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models,params)

           best_model_score = max(sorted(model_report.values()))
           best_model_name = [key for key,value in model_report.items() if value == best_model_score]

           best_model = models[best_model_name[0]]

           if best_model_score < 0.6:
               raise CustomException("No best model found")
           
           logging.info("Best model found on our dataset")

           save_object(file_path = self.model_trainer_config.trained_model_file_path,obj = best_model)

           predicted = best_model.predict(x_test)
           r2_sco = r2_score(y_test,predicted)
           return r2_sco

        except Exception as e:
            raise CustomException(e,sys)

