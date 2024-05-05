## Description
This repository aims to provide tools for comparing different explainability methods, enhancing the interpretation of machine learning models. It currently includes demonstrations of SHAP and LIME, with the intention of expanding to include more interpretability techniques in the future.

### SHAP
- SHAP values provide global interpretations of a model's output by attributing each feature's contribution to the predicted outcome.
- The script initializes a RandomForestRegressor model and explains its global behavior using SHAP.

### LIME
- LIME provides local interpretations of individual predictions by approximating the model's behavior around specific data points.
- The script initializes a LimeTabularExplainer and explains local predictions of the RandomForestRegressor model using LIME.

## Status
This repository is under construction. Additional features and interpretability methods will be added in future updates.

## File Structure
- `main.py`: Main Python script demonstrating the usage of SHAP and LIME for model interpretability.
- `LIME.py`: Wrapper class for LIME explanations.
- `SHAP.py`: Wrapper class for SHAP explanations.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The California housing dataset is sourced from scikit-learn.
- SHAP and LIME libraries are used for model interpretability.
