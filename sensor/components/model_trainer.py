from sensor.entity import config_entity,artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
from typing import Optional
import pandas as pd
import numpy as np
import os,sys
from sensor import utils
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

class ModelTrainer:
    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            
           logging.info(f"{'>>'*20} model trainer {'<<'*20}")
           self.model_trainer_config=model_trainer_config
           self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise SensorException(e,sys)
   
    def train_model(self,x,y):
        xgb_clf=XGBClassifier()
        xgb_clf.fit(x,y)
        return xgb_clf

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f" Loading train & test array")
            train_arr=utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr=utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)
            
            logging.info(f"Splitting the input and target fetaure from both train & test arr")
            x_train,y_train=train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test=test_arr[:,:-1],test_arr[:,-1]
            
            logging.info(f'train the model')
            model=self.train_model(x=x_train,y=y_train)
           
            logging.info(f'calculating f1 train score')
            yhat_train=model.predict(x_train)
            f1_train_score=f1_score(y_pred=yhat_train,y_true=y_train)
            
           
            logging.info(f'calculating f1 test score')
            yhat_test=model.predict(x_test)
            f1_test_score=f1_score(y_true=y_test,y_pred=yhat_test)

            logging.info(f"train score:{f1_train_score} and test score {f1_test_score}")

            # Checking for overfitting or under fitting or expected score
           
            logging.info(f"Checking p=our model is under fitting or not")
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Moddel is not good as it is not able to give\
                expected accuracy:{self.model_trainer_config.expected_score}:model actual score:{f1_test_score}")

            logging.info(f"Checking for if our model is overfitting or not")
            diff=abs(f1_train_score-f1_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
               raise Exception(f"Train & test score diff:{diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")
            # save the trained model
            logging.info(f" saving model ogject")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            # prepare the artifact
            model_trainer_artifact=artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, f1_train_score=f1_train_score,f1_test_score=f1_test_score)
            logging.info(f'model trainer artifact:{model_trainer_artifact}')
            return model_trainer_artifact

        except Exception as e:
            raise SensorException(e,sys)  
