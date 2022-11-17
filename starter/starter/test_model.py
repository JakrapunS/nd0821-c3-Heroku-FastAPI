"""
Unit test for ml model
"""

import pickle
import pandas as pd

model = pickle.load(open('/home/jakrapun/Heroku/nd0821-c3-Heroku-FastAPI/starter/model/ML_model.pkl','rb'))

print(model.get_params())
def test_model():
    model = pickle.load(open('/starter/model/ML_model.pkl','rb'))
    assert model.get_params()['max_iter'] == 300

