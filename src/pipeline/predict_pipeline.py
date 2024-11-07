import pandas as pd
from src.exception import CustomException
import sys
import os
from src.utils import load_object

class CustomData:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,
                 lunch,test_preparation_course,reading_score,writing_score):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def get_data_as_data(self):
        try:
            Custom_dict={"gender":[self.gender],
                        "race ethnicity":[self.race_ethnicity],
                        "parental level of education":[self.parental_level_of_education],
                        "lunch":[self.lunch],
                        "test preparation course":[self.test_preparation_course],
                        "reading score":[self.reading_score],
                        "writing score":[self.writing_score]}
            return pd.DataFrame(Custom_dict)
        except Exception as e:
            raise CustomException(e,sys)

class PredictPipeline:
    def Predict_Data(self,features):
        try:
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            feature = pd.DataFrame(features,columns=["gender","race/ethnicity","parental level of education","lunch","test preparation course","reading score","writing score"])
            if isinstance(feature, pd.DataFrame):
                print(type(feature))
                print(feature)
                data_scaled = preprocessor.transform(feature)
            else:
                print(type(feature))
                raise ValueError("Input features must be a Pandas DataFrame.")
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e,sys)
