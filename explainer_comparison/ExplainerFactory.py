# ----------------------------------------------------------------------------------------------------
# Class ExplainerFactory
# This class is used to create Explainer objects based on what word is entered into the create() function
#
# ------------------------------------------------------------------------------------------------------

import string
from typing import Any

import pandas as pd
import xgboost as xgb

from Explainer import Explainer
from LIME import LIME
from SHAP import SHAP
from constants import MODE


class ExplainerFactory:
    # If user wants to use XGBoost explanation, model, X, and y must be filled in as parameters for init
    # all possible parameters are set to None as default.
    
    def __init__(self,
                 model: Any = None,
                 X_train: pd.DataFrame = None,
                 X_test: pd.DataFrame = None,
                 y_train: pd.DataFrame = None,
                 y_test: pd.DataFrame = None,
                 mode: str = MODE.REGRESSION):  # Default to REGRESSION
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.mode = mode


    def create_explainer(self, explainer_type: string) -> Explainer:
        if explainer_type == "shap":
            shapEx = SHAP(self.model, self.X_train, self.y_train, mode=self.mode)
            return shapEx
        elif explainer_type == "lime":
            limeEx = LIME(self.model, self.X_train, self.y_train, mode=self.mode)
            return limeEx
        #elif explainer_type == "xgboost":
        #    return self.create_xgb_global_feature_importance(self.model, self.X, self.y)

        # If there are more explainers you want to account for, the code can be added here:

        else:
            # throw exception
            print("invalid Explainer")

    # Check to see if you can restrict the type of the model and output.
    #def create_xgb_global_feature_importance(self,
    #                                         model: Any,
    #                                         X: pd.DataFrame,
    #                                         y: pd.DataFrame) -> Any:
    #    if not isinstance(model, xgb.XGBClassifier):
    #        raise ValueError("model must be an XGBoost model")
    #    elif not isinstance(model, xgb.XGBRegressor):
    #        raise ValueError("model must be an XGBoost model")
    #    model.fit(X, y)
    #    return model.feature_importances_
