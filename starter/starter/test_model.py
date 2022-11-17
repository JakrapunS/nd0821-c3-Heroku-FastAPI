"""
Unit test for ml model
"""

import pickle
import pandas as pd


def test_model():
    model = pickle.load(open('/starter/model/ML_model.pkl','rb'))
    assert model.get_params()['max_iter'] == 300

