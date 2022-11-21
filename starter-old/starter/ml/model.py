from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle5 as pickle
import pandas as pd
import numpy as np

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """


    model = RandomForestClassifier(n_estimators=100,min_samples_split=10, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model,name):
    pickle.dump(model, open(f"nd0821-c3-Heroku-FastAPI/starter/model/{name}", 'wb'))



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.  
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    preds = model.predict(X)
    return preds

def slice_matrix(test_data,Y_pred,features):
    """ 
    Calculate matrix through slice data by categorical features and save as csv to Slicer_performance folder.

    Inputs
    ------
    test_data : pd.DataFrame
        Test dataframe.  
    Y_pred : list
        Prediction from model.
    features : np.list
    
    """

    df = test_data.reset_index().copy()
    
    df['pred'] = pd.Series(np.squeeze(Y_pred))

    for feature in features:
        TP = df[df['salary'] == ">50K"].groupby(feature)['pred'].sum()
        FP = df[df['salary'] == ">50K"].groupby(feature)['pred'].apply(lambda x: x.count() - x.sum())
        TN = df[df['salary'] == "<=50K"].groupby(feature)['pred'].apply(lambda x: x.count() - x.sum())
        FN = df[df['salary'] == "<=50K"].groupby(feature)['pred'].sum()


        precision = (TP / (TP + FP))
        recall = (TP / (TP + FN))
        TNR = (TN / (TN + FP))  # True Negative Rate
        NPV = (TN / (TN + FN))  # Negative Predictive Value
        f_score = 2*((precision * recall) / (precision + recall))

        slice_performance = pd.concat([precision, recall, TNR, NPV, f_score], axis=1)
        slice_performance.columns = ['Precision', 'Recall', 'TNR', 'NPV', 'F-Score']
        slice_performance.to_csv(f"/home/jakrapun/Heroku/nd0821-c3-Heroku-FastAPI/starter/Slicer_performance/slice_performance_{feature}.csv")

    

    #return slice_performance



