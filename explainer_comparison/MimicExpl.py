# ----------------------------------------------------------------------------------------------------
# Class Explainable Boosting Machine
#
# This class wraps the interpret.glassbox.ExplainableBoostingRegressor and ExplainableBoostingClassifier explainer methods
#
# ------------------------------------------------------------------------------------------------------
from interpret.ext.blackbox import MimicExplainer

# You can use one of the following four interpretable models as a global surrogate to the black box model
from interpret.ext.glassbox import LGBMExplainableModel
#from interpret.ext.glassbox import LinearExplainableModel
#from interpret.ext.glassbox import SGDExplainableModel
#from interpret.ext.glassbox import DecisionTreeExplainableModel
import numpy as np
import pandas as pd


from Explainer import Explainer

# # Handle it later
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")


class MimicExpl(Explainer):

    def __init__(self, model, X_train, y_train, y_pred=None, mode='regression'):
        super().__init__(model, X_train, y_train, y_pred, mode)
        self.create_explainer()

    def create_explainer(self):
        self.explainer = MimicExplainer(self.model, 
                           self.X_train, 
                           LGBMExplainableModel, 
                           augment_data=True, 
                           max_num_of_augmentations=10)
                        #    features=data.feature_names, 
                        #    classes=data.target_names.tolist())
        
        # self.explainer.fit(self.X_train, self.y_train)
        return self.explainer
    
    def predict(self, X_data: pd.DataFrame) -> pd.DataFrame:
        y_pred = self.explainer._get_surrogate_model_predictions(X_data)

        y_pred = pd.Series(y_pred).replace(to_replace=['malignant', 'benign'], value=[0, 1]).to_numpy()

        # return self.explainer.predict(X_data)
        return y_pred
    
    def predict_proba(self, X_data: pd.DataFrame) -> pd.DataFrame:
        return np.zeros(shape=len(X_data), dtype=float)
        # if self.mode != 'regression':
        #     return self.explainer.predict_proba(X_data)
        # else:
        #     raise NotImplementedError('predict_proba is not available for the regression mode')



    def explain_global(self, X_data: pd.DataFrame) -> pd.DataFrame:
        return self.explainer.explain_global()


    def explain_local(self, X_data: pd.DataFrame) -> pd.DataFrame:
        return self.explainer.explain_local(X_data)


