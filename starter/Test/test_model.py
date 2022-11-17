"""
Unit test for ml model
"""
import pickle
import os

model = pickle.load(open("/home/jakrapun/Heroku/nd0821-c3-Heroku-FastAPI/starter/model/RandomForest.pkl", 'rb'))

def test_n_class():
    model = pickle.load(open(os.path.join('./', "nd0821-c3-Heroku-FastAPI/starter/model/ML_model.pkl"), 'rb'))
    assert model.n_classes_ == 2

def test_min_sample_leaf():
    model = pickle.load(open(os.path.join('./', "nd0821-c3-Heroku-FastAPI/starter/model/ML_model.pkl"), 'rb'))
    assert model.get_params()['min_samples_leaf'] == 5

def test_min_samples_split():
    model = pickle.load(open( os.path.join('./', "nd0821-c3-Heroku-FastAPI/starter/model/ML_model.pkl"),'rb'))

    assert model.get_params()['min_samples_split'] == 10
