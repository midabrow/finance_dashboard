
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import numpy as np


def metrics_calculator(y_pred, y_test, model_type, model_name, average='macro'):
    """
    Calculates core evaluation metrics (Accuracy, Precision, Recall, F1-score) for a classification model
    or (MSA, MSE, RMSE, R2 Socre) for regression model and returns them as a pandas DataFrame.

    Parameters:
    -----------
    y_pred : array-like
        The predicted class labels from the model.

    y_test : array-like
        The true class labels from the test set.

    model_type: str
        Name of the model type - `Regression` or `Classification`

    model_name : str
        Name of the model to be used as the column header in the output DataFrame.

    average : str, default='macro'
        The averaging method for multi-class classification.
        Common options include 'binary', 'macro', 'micro', or 'weighted'.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing (Accuracy, Precision, Recall, and F1-score) or (MSA, MSE, RMSE, R2 Socre),
        indexed by metric name, and with model name as the column.

    Raises:
    -------
    ValueError:
        If either y_pred or model_name is None.

    Notes:
    ------
    - This function is useful for benchmarking and comparing multiple models.
    - To build a complete comparison table, you can concatenate the outputs
      from multiple models using pd.concat().
    """
    if y_pred is None or model_name is None:
        raise ValueError("Prediction or model name is None. Check your input values.")
    
    if model_type == 'Classification':
        result = pd.DataFrame(data=[accuracy_score(y_test, y_pred),
                                    precision_score(y_test, y_pred, average=average, zero_division=0),
                                    recall_score(y_test, y_pred, average=average, zero_division=0),
                                    f1_score(y_test, y_pred, average=average, zero_division=0)],
                            index=['Accuracy','Precision','Recall','F1-score'],
                            columns=[str(model_name)])
        
        result = (result * 100).round(2).astype(str) + '%'  

    elif model_type == 'Regression':
        result = pd.DataFrame(data=[mean_absolute_error(y_test, y_pred),
                                    mean_squared_error(y_test, y_pred),
                                    np.sqrt(mean_squared_error(y_test, y_pred)),
                                    r2_score(y_test, y_pred)],
                            index=['MAE', 'MSE', 'RMSE' 'R2 Score'],
                            columns=[str(model_name)])
        
        result = result.round(3)

    else:
        raise ValueError("There is not such a type of model")
    
    print(f"Model {model_name} metrics computed successfully")
    
    return result