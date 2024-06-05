# ----------------------------------------------------------------------------------------------------
# Class LIME
#
# This class wraps the lime explainer method
#
# ------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import lime
from typing import List, Optional, Callable

from Explainer import Explainer
from constants import MODE

# Handle it later
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")


class LIME(Explainer):

    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.DataFrame, y_pred: pd.DataFrame = None, mode: str = 'regression'):
        super().__init__(model, X_train, y_train, y_pred, mode)
        self.explainer = None
        self.class_names = list(y_train.unique())
        print(self.class_names)
        self.create_explainer()


    def create_explainer(self):
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.X_train.columns,
            class_names=self.class_names if self.mode == MODE.CLASSIFICATION else None,
            mode=self.mode
        )


    def explain_local(self, X_data: pd.DataFrame) -> pd.DataFrame:
        if self.explainer is None:
            self.create_explainer()
        print('Creating LIME explanation')
        lime_coefficients = np.zeros(shape=X_data.reset_index(drop=True).shape, dtype=float)
        for i, row in X_data.reset_index(drop=True).iterrows():
            exp = self.explainer.explain_instance(
                data_row=row, 
                predict_fn=self.model.predict_proba if self.mode == MODE.CLASSIFICATION else self.model.predict,
                num_features=len(X_data.columns),
                labels=self.class_names,
                num_samples=100
                )
            sorted_exps = sorted(exp.local_exp[1])
            slope_coefs = np.array([expl[1] for expl in sorted_exps])
            lime_coefficients[i] = slope_coefs

            # Progress bar
            progress = i / X_data.shape[0]
            print(f"\r[{'='*int(progress*100):<100}] {progress*100:.2f}%", end="")
        
        return pd.DataFrame(lime_coefficients)


    def explain_global(self, X_data: pd.DataFrame) -> pd.DataFrame:
        local_exps = self.explain_local(X_data)
        # Calculate the mean across rows to get the average effect of each feature globally
        global_exp = np.mean(local_exps.values * X_data.values, axis=0)
        # Convert to DataFrame to match the requested output format
        global_explanations = pd.DataFrame(global_exp, index=X_data.columns, columns=['LIME Value'])
        return global_explanations
    

    def predict(self, X_data: pd.DataFrame):
        """
        Predict class labels for the input data using LIME explanations.
        """
        # Get predicted probabilities
        lime_pred_estimation = self.predict_proba(X_data)
        # Convert predicted probabilities to class labels
        y_pred_class_lime = np.where(lime_pred_estimation > 0.5, 1, 0)

        return y_pred_class_lime


    def predict_proba(self, X_data: pd.DataFrame):
        """
        Predict probabilities for the input data using LIME explanations.
        """
        if self.mode == MODE.REGRESSION:
            raise NotImplementedError('predict_proba is not available for the regression mode')
        else:

            num_rows, num_features = X_data.shape
            # Initialize an array to store the predicted probabilities
            lime_pred_estimation = np.zeros(shape=num_rows, dtype=float)
            # Iterate over each instance in the input data
            for i, instance in X_data.reset_index(drop=True).iterrows():
                exp = self.explainer.explain_instance(instance.values, predict_fn=self.model.predict_proba, labels=self.class_names, num_features=num_features)
                # Store the local prediction
                print(exp.local_pred)
                print(exp.predict_proba)
                lime_pred_estimation[i] = exp.local_pred.item()
            # Clip the predicted probabilities to be within the range [0, 1]
            lime_pred_estimation = np.clip(lime_pred_estimation, 0, 1)

            return lime_pred_estimation