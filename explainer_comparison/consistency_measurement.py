import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
from explainer_comparison.ExplainerFactory import ExplainerFactory
from explainer_comparison.explainer_utilities import run_and_collect_explanations
from tqdm import tqdm

def visualize_consistency(explainers, feature_names, summary):
    num_explainers = len(explainers)
    fig, axes = plt.subplots(1, num_explainers, figsize=(15, 6), sharey=True)

    # Visualization
    for ax, explainer in zip(axes, explainers):
        mean_impact, std_impact = summary[explainer]
        ax.barh(feature_names, mean_impact, xerr=std_impact, align='center', alpha=0.7, ecolor='black', capsize=5)
        ax.set_xlabel('Mean Feature Impact')
        ax.set_title(f'Feature Impact - {explainer.upper()}')

    plt.tight_layout()
    plt.show()


def consistency_measurement(X, y, model, n_splits=5, explainers=None, verbose=False):

    folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    available_explainers = ["shap", "lime"] #, "ebm"] #, "mimic"]  # Easily extendable for additional explainers
    feature_names = X.columns

    chosen_explainers = explainers if explainers is not None else available_explainers
    results = {explainer: [] for explainer in chosen_explainers}

    # Train the model on the full dataset first
    model.fit(X, y)
    
    for train_index, test_index in tqdm(folds.split(X, y), total=n_splits, desc="Processing folds"):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        factory = ExplainerFactory(model, X_train, X_test, y_train, y_test)
        explanations = run_and_collect_explanations(factory, X_train, explainers=chosen_explainers, verbose=verbose)

        for explainer in chosen_explainers:
            explainer_values = explanations[explainer.upper() + " Value"]
            results[explainer].append(explainer_values)

    summary = {}
    for explainer in explainers:
        results[explainer] = np.array(results[explainer])
        mean_impact = np.mean(results[explainer], axis=0)
        std_impact = np.std(results[explainer], axis=0)
        summary[explainer] = (mean_impact, std_impact)

    # Visualization
    visualize_consistency(chosen_explainers, feature_names, summary)

    final_scores = {explainer: {'min_std': np.min(summary[explainer][1]), 
                                'max_std': np.max(summary[explainer][1]),
                                'mean_std': np.mean(summary[explainer][1]), 
                                'median_std': np.median(summary[explainer][1])} for explainer in explainers}
    final_scores_df = pd.DataFrame(final_scores).T
    
    # Display side by side final scores of all XAI methods
    print(final_scores_df)

    return summary, final_scores_df