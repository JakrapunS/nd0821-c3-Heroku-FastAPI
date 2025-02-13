# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, save_model,slice_matrix
from sklearn.metrics import classification_report
# Add code to load in the data.
data = pd.read_csv('/home/jakrapun/Heroku/nd0821-c3-Heroku-FastAPI/starter/data/census.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20,random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,encoder=encoder,lb=lb
)
# Train and save a model.
model = train_model(X_train, y_train)

save_model(model,"RandomForest.pkl")
save_model(encoder, "encoder.pkl")
save_model(lb,  "label_binarizer.pkl")

pred = model.predict(X_test)

slice_performance = slice_matrix(test,pred,cat_features)

print("------- Random Forest test set performance -------")
print(classification_report(y_test, pred))