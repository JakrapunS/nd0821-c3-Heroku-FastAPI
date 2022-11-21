# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Random Forest model was used as classification model for salary census data.
* Parameter: n_estimators=100, min_samples_split=10, min_samples_leaf=5
* Random state = 42
## Intended Use
This model is intended to be used for predict that salary of specific person will be higher 50k or lower than 50k based on demographical data and geographical data.
## Training Data
80% of Census Income Data Set (https://archive.ics.uci.edu/ml/datasets/census+income) was used for create training set.
## Evaluation Data
20% of Census Income Data Set (https://archive.ics.uci.edu/ml/datasets/census+income) was used for create test set.
## Metrics
* Macro F1-Score: 0.80
* Accuracy: 0.86

## Ethical Considerations
The model was trained on Census Income Data Set which is small dataset which is donated in 1996. This mean the data potentially outdated and might not able to use to predict income of recent year. Moreover, performance on slicer data show potentially bias on gender, occupation and country.

## Caveats and Recommendations
This model parameters did not yet optimized. Moreover, this model did not compare with other type of models. Therefore, I would suggest to try another type of model and compare performance across models.