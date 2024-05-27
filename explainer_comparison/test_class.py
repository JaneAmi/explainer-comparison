# ----------------------------------------------------------------------------------------------------
# file for testing printing Lime and SHAP for binary classification
#
#
# ------------------------------------------------------------------------------------------------------

# Standard library imports
import pandas as pd
import numpy as np

# Third party imports
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_tabular

# Local application imports
from ExplainerFactory import ExplainerFactory
from explainer_utilities import run_and_collect_explanations


# Load the Breast Cancer dataset
data = load_breast_cancer()

# Create a DataFrame with feature names
X = pd.DataFrame(data.data, columns=data.feature_names)

# Create a Series for the target variable
y = pd.Series(data.target, name='target')


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




# Create a random forest model then train it
print('\n Training a model ...')
# Initialize and train the XGBoost Classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)
print('\n Model trained')

# Initialize the ExplainerFactory with the trained model and data splits.

expl_fctry = ExplainerFactory(model, X_train, X_test, y_train, y_test)

results = run_and_collect_explanations(expl_fctry, X_test)
print(results)

def plot_lime_shap(data, shap_column, lime_column):
    
    colors = sns.color_palette("deep")
    plt.figure(figsize=(8, 6))    
    
    bar_positions = np.arange(len(data))  # Positions of the bars
    bar_width = 0.35  # Bar widths

    plt.barh(bar_positions - bar_width/2, data[shap_column], height=bar_width, label='SHAP', color=colors[0])  # PSHAP values
    plt.barh(bar_positions + bar_width/2, data[lime_column], height=bar_width, label='LIME', color=colors[1])  # LIME values
    plt.yticks(bar_positions, data.index)  #labels

    plt.title('Feature Importances from SHAP and LIME')
    plt.legend()
    plt.show()

plot_lime_shap(results, 'SHAP Value', 'LIME Value')
print('\n plotting completed')