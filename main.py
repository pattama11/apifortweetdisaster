# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 11:36:48 2022

@author: siddhardhan
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    
    Pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction :  float
    Age : int
    

# loading the saved model
disaster_model = pickle.load(open('disaster_classifier.sav','rb'))


@app.post('/disaster_classifier')
def disaster_pred(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    text = 'ไฟไหม้'
    location = input_dictionary['ลาดกระบัง']

    input_list = [text,location]
    
    prediction = disaster_model.predict(text,k=4)
    
    if prediction[0] == 'fire':
        return 'fire' + 'in' + location
    else:
        return 'normal'


