# ----------------------------------------------------------------------------------------------------
# Class Explainable Boosting Machine
#
# This class wraps the interpret.glassbox.ExplainableBoostingRegressor and ExplainableBoostingClassifier explainer methods
#
# ------------------------------------------------------------------------------------------------------
import interpret.glassbox
import numpy as np
import pandas as pd

from Explainer import Explainer

# # Handle it later
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")


class EBM(Explainer):

    def __init__(self, model, X_train, y_train, y_pred=None, mode='regression'):
        super().__init__(model, X_train, y_train, y_pred, mode)
        self._create_explainer()

    def _create_explainer(self):

        if self.mode == 'regression':
            self.explainer = interpret.glassbox.ExplainableBoostingRegressor()
        else:
            self.explainer = interpret.glassbox.ExplainableBoostingClassifier()
        
        self.explainer.fit(self.X_train, self.y_train)

    
    def predict(self, X_data: pd.DataFrame) -> pd.DataFrame:
        return self.explainer.predict(X_data)
    
    def predict_proba(self, X_data: pd.DataFrame) -> pd.DataFrame:

        if self.mode != 'regression':
            return self.explainer.predict_proba(X_data)
        else:
            raise NotImplementedError('predict_proba is not available for the regression mode')



    def explain_global(self, X_data: pd.DataFrame) -> pd.DataFrame:
        return self.explainer.explain_global()


    def explain_local(self, X_data: pd.DataFrame) -> pd.DataFrame:
        return self.explainer.explain_local(X_data)


