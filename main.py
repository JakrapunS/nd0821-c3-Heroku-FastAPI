#Put the code for your API here.
from fastapi import FastAPI
from typing import Optional, Union
from pydantic import BaseModel, Field
import pandas as pd
import pickle

import sys



sys.path.insert(0, '/home/jakrapun/Heroku/nd0821-c3-Heroku-FastAPI/starter')

from starter.starter.ml.model import inference
from starter.starter.ml.data import process_data

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

class Item(BaseModel):
    age: Optional[Union[int, list]] = [39, 31]
    workclass: Optional[Union[str,list]] = ["State-gov","Private"]
    fnlgt: Optional[Union[int,list]] = [77516, 45781]
    education: Optional[Union[str, list]] = ['Bachelors', 'Masters']
    education_num: Optional[Union[int, list]] = Field([13, 14], alias='education-num')
    marital_status: Optional[Union[str, list]] = Field(['Never-married', 'Never-married'], alias='marital-status')
    occupation: Optional[Union[str, list]] = ['Adm-clerical', 'Prof-specialty']
    relationship: Optional[Union[str, list]] = ['Not-in-family', 'Not-in-family']
    race: Optional[Union[str, list]] = ['White', 'White']
    sex: Optional[Union[str, list]] = ['Male', 'Female']
    capital_gain: Optional[Union[int, list]] = Field([2174, 14084], alias='capital-gain')
    capital_loss: Optional[Union[int, list]] = Field([0, 0], alias='capital-loss')
    hours_per_week: Optional[Union[int, list]] = Field([40, 50], alias='hours-per-week')
    native_country: Optional[Union[str, list]] = Field(['United-States', 'United-States'], alias='native-country')


class Config:
    allow_population_by_field_name = True

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"Greeting": "Welcome to My ML project!"}
 
@app.post("/model_inference")
async def inference_post(data: Item):
    data_dict = data.dict(by_alias=True)
    
    for key, value in data_dict.items():
        data_dict[key] = [value]

    

     
    df = pd.DataFrame(data_dict)

    
    #path = 'starter/model/RandomForest.pkl'
    path = '/home/jakrapun/Heroku/nd0821-c3-Heroku-FastAPI/starter/model/RandomForest.pkl'
    model = pickle.load(open(path,'rb'))
    #path_encoder = 'starter/model/encoder.pkl'
    path_encoder= '/home/jakrapun/Heroku/nd0821-c3-Heroku-FastAPI/starter/model/encoder.pkl'

    encoder = pickle.load(open(path_encoder,'rb'))

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    processed_df, _, _, _ = process_data(
        df, categorical_features=categorical_features, label=None, training=False,
        encoder=encoder, lb=None
    )

    pred = list(inference(model, processed_df))

    for idx, val in enumerate(pred):
        if pred[idx] == 0:
            pred[idx] = '<=50K'
        else:
            pred[idx] = '>50K'

    return {"Result": pred}

    