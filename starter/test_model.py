"""
Unit test for ml model
"""

import pickle5 as pickle
import pandas as pd
import os


def test_model_min_leaf():
    path = 'starter/model/RandomForest.pkl'
    model = pickle.load(open(path,'rb'))

    assert model.get_params()['min_samples_leaf'] == 5
    
def test_model_min_split():
    path = 'starter/model/RandomForest.pkl'
    model = pickle.load(open(path,'rb'))

    assert model.get_params()['min_samples_split'] == 10

def test_model_n_estimators():
    path = 'starter/model/RandomForest.pkl'
    model = pickle.load(open(path,'rb'))

    assert model.get_params()['n_estimators'] == 100
