"""
Unit test for ml model
"""

import pickle
import pandas as pd
import os



def test_model():
    path = 'starter/model/ML_model.pkl'
    model = pickle.load(open(path,'rb'))

    assert model.get_params()['max_iter'] == 300
    

