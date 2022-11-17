"""
Unit test for ml model
"""

import os
from ml.model import inference
import pandas as pd





def test_inference(model, data):
    X, y = data
    y_pred = inference(model, X)

    assert len(y_pred) == len(y)
    assert y_pred.any() == 1

