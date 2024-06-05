from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

from explainers import EBM, MimicExpl, LIME
from explainer_utilities import create_metrics_dataframe
from ExplainerFactory import ExplainerFactory

# Load the Breast Cancer dataset
data = load_wine()

# Create a DataFrame with feature names
X = pd.DataFrame(data.data, columns=data.feature_names)

# Create a Series for the target variable
y = pd.Series(data.target, name='target')


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_test = X_test.iloc[:10,:]

# Initialize and train the XGBoost Classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

model_pred_proba = model.predict_proba(X_test)[:,1]

predict_function = model.predict
model_pred_class = predict_function(X_test).values \
    if type(predict_function(X_test)) in [pd.DataFrame, pd.Series] \
    else predict_function(X_test)



lime = LIME.LIME(model, X_train, y_train, mode='classification')
lime_pred_proba = lime.predict_proba(X_test)
lime_pred_class = lime.predict(X_test)


import lime_x
explainer = lime_x.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns,
            class_names=list(y_train.unique()),
            mode='classification'
        )

instance = X_test.to_numpy()[0]
num_features = X_test.shape[1]

exp = explainer.explain_instance(instance, predict_fn=model.predict_proba, labels=list(y_train.unique()), num_features=num_features)
print(exp.local_pred)