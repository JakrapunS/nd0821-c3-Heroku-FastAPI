from fastapi.testclient import TestClient
import pandas as pd
from main import app

client = TestClient(app)

def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Greeting": "Welcome to My ML project!"}

def test_pos_class():
    r = client.post("/model_inference")
    data = pd.read_csv('/home/jakrapun/Heroku/nd0821-c3-Heroku-FastAPI/starter/data/census.csv')
    data = data.astype(str)
    X = data.iloc[10]
    y = X['salary']
    X.drop('salary')

    r = client.post("/model_inference", json=X.to_dict())

    assert r.status_code == 200
    assert r.json() == {'Prediction': ['>50K']}

def test_neg_class():
    r = client.post("/model_inference")
    data = pd.read_csv('/home/jakrapun/Heroku/nd0821-c3-Heroku-FastAPI/starter/data/census.csv')
    data = data.astype(str)
    X = data.iloc[0]
    y = X['salary']
    X.drop('salary')

    r = client.post("/model_inference", json=X.to_dict())

    assert r.status_code == 200
    assert r.json() == {'Prediction': ['<=50K']}


