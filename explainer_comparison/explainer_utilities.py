import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, precision_score, recall_score
from ExplainerFactory import ExplainerFactory

def run_and_collect_explanations(factory: ExplainerFactory, X_data, explainers=None) -> pd.DataFrame:
    results = []
    available_explainers = ["shap", "lime"]  # Easily extendable for additional explainers
    
    chosen_explainers = explainers if explainers is not None else available_explainers

    for explainer_type in chosen_explainers:
        explainer = factory.create_explainer(explainer_type)
        if explainer is not None:
            try:
                global_explanation = explainer.explain_global(X_data)
                results.append(global_explanation)
                print(f'\n {explainer_type.upper()} explanation created')
            except Exception as e:
                print(f"Failed to create {explainer_type.upper()} explanation: {e}")
        else:
            print(f"No explainer available for type: {explainer_type}")

    # Concatenate all results along columns (axis=1), handling cases where some explanations might fail
    if results:
        return pd.concat(results, axis=1)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no explanations were added
    


def create_metrics_dataframe(estimator_names, pred_estimations, y_pred_classes, model_pred, model_pred_class):
    """
    Creates a DataFrame with evaluation metrics for given estimators.

    Parameters:
    estimator_names (list): List of estimator names.
    pred_estimations (list): List of predicted estimations for each estimator.
    y_pred_classes (list): List of predicted classes for each estimator.
    model_pred (array-like): Model's predicted values.
    model_pred_class (array-like): Model's predicted classes.

    Returns:
    pd.DataFrame: DataFrame with evaluation metrics.
    """
    metrics = {
        'MSE': [mean_squared_error(pred, model_pred) for pred in pred_estimations],
        'ACCURACY': [accuracy_score(y_pred, model_pred_class) for y_pred in y_pred_classes],
        'F1_SCORE': [f1_score(y_pred, model_pred_class) for y_pred in y_pred_classes],
        'PRECISION': [precision_score(y_pred, model_pred_class) for y_pred in y_pred_classes],
        'RECALL': [recall_score(y_pred, model_pred_class) for y_pred in y_pred_classes]
    }

    results = pd.DataFrame(metrics, index=estimator_names).T
    return results


