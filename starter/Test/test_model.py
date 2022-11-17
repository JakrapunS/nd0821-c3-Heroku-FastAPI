"""
Unit test for ml model
"""
from starter.train_model import model
import os

def test_n_class(model):
    
    assert model.n_classes_ == 2

def test_min_sample_leaf(model):
    
    assert model.get_params()['min_samples_leaf'] == 5

def test_min_samples_split(model):

    assert model.get_params()['min_samples_split'] == 10
