import requests
import json
import sys
import pandas as pd

#sys.path.insert(0, '/home/jakrapun/Heroku/nd0821-c3-Heroku-FastAPI/starter')



data = pd.read_csv('starter/data/census.csv')
data = data.astype(str)



X = data.iloc[10]
y = X['salary']

X.drop('salary')

response = requests.post('http://127.0.0.1:8000/model_inference',
                         data=json.dumps(X.to_dict()))
                         
print(f"Status Code: {response.status_code}")
print(f"Ground Truth: {y}, Response: {response.json()}")

