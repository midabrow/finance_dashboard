import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from yellowbrick.classifier import ConfusionMatrix, ROCAUC, PrecisionRecallCurve
from yellowbrick.model_selection import LearningCurve, FeatureImportances
from yellowbrick.contrib.wrapper import wrap
from yellowbrick.style import set_palette

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error

import plotly.graph_objs as go
import plotly.colors as colors
from plotly.subplots import make_subplots

from helpers import config as cf
from helpers import data_visualization_utils as viz
from helpers import metrics_utils as met

# ============================================================
# 🔹 BALANCE CHECKING
# ============================================================

def check_balance_in_data(df, column):
    return round(df[column].value_counts(normalize=True) * 100, 0)


def plot_target_balance(df, target_column, chart_type="bar"):
    """
    Funkcja do wizualizacji balansu targetu.
    
    :param df: DataFrame zawierający dane
    :param target_column: Nazwa kolumny targetu
    :param chart_type: Rodzaj wykresu - "bar" (domyślnie) lub "pie"
    """
    # Zliczanie wartości unikalnych
    target_counts = df[target_column].value_counts().sort_index()
    
    # Kolory dla lepszej czytelności
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]

    if chart_type == "bar":
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=target_counts.index.astype(str), 
            y=target_counts.values, 
            marker_color=colors[:len(target_counts)],
            text=target_counts.values, 
            textposition='auto'
        ))
        fig.update_layout(title=f"Balans wartości {target_column}",
                          xaxis_title=target_column,
                          yaxis_title="Liczność",
                          template="plotly_dark")

    elif chart_type == "pie":
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=target_counts.index.astype(str), 
            values=target_counts.values,
            marker=dict(colors=colors[:len(target_counts)]),
            textinfo="label+percent"
        ))
        fig.update_layout(title=f"Procentowy udział {target_column}", template="plotly_dark")

    else:
        raise ValueError("chart_type powinien być 'bar' lub 'pie'")

    fig.show()


# ============================================================
# 🔹 MODEL COMPARISON
# ============================================================

def run_model_comparison(models, X_train, y_train, X_test, y_test, n_splits=5, problem_type='Classification'):
    """
    Compares multiple models using GridSearchCV, evaluates them, and returns performance metrics.

    Parameters:
    -----------
    models : list of dict
        A list of dictionaries, each containing:
        - 'name': model name (str)
        - 'model': model class (e.g., RandomForestClassifier)
        - 'params': dictionary of hyperparameters
    
    X_train, y_train : array-like
        Training data and labels.

    X_test, y_test : array-like
        Test data and labels.

    n_splits : int, default=5
        Number of cross-validation folds used in GridSearchCV.

    problem_type : str, default='Classification'
        Type of machine learning task: 'Classification' or 'Regression'.

    Returns:
    --------
    pd.DataFrame
        A summary DataFrame of all models with:
        - training and test accuracy
        - best hyperparameters
        - precision, recall, and F1-score (for classification)
    
    Notes:
    ------
    - Internally uses the `train_and_evaluate()` function to fit each model and collect results.
    - Ideal for comparing baseline models during the model selection phase.
    - Extendable: you can add ROC AUC, log loss, or training time in the results.
    """

    results = []

    for model_info in models:
        model_name = model_info['name']
        model_class = model_info['model']
        params = model_info['params']
        
        print(f"\n{cf.clr.orange}{'='*20} Model testing: {model_name} {'='*20}{cf.clr.reset}\n")

        # Trenowanie modelu z najlepszymi hiperparametrami
        # best_model, train_score, test_score, metrics_results = 
        fit_and_evaluate_model(
            model_class, params, X_train, y_train, X_test, y_test, 
            n_splits=n_splits, problem_type=problem_type
        )

    #     results.append({
    #         'model': model_name,
    #         'train_score': train_score,
    #         'test_score': test_score,
    #         'best_params': best_model.get_params(),
    #         'precision': metrics_results.loc['Precision', model_class.__name__],
    #         'recall': metrics_results.loc['Recall', model_class.__name__],
    #         'F1-score': metrics_results.loc['F1-score', model_class.__name__]
    #     })

    # df_results = pd.DataFrame(results)
    # df_results = df_results.set_index('model').T
    # return df_results





def fit_and_evaluate_model(model_class, params, X_train, y_train, X_test, y_test=None, n_splits=3, show_feature_importance=False, problem_type='Regression', plot_metrics=False, binary=True):
    """
    Trains and evaluates a model using GridSearchCV (for classification) or K-Fold CV (for regression).
    Returns evaluation scores and the trained model.

    Parameters:
    -----------
    model_class : class
        The machine learning estimator class (e.g., RandomForestClassifier).

    params : dict
        Hyperparameter grid for GridSearchCV (for classification) or model instantiation (for regression).

    X_train, y_train : array-like
        Training data and labels.

    X_test, y_test : array-like
        Test data and labels. y_test can be None.

    n_splits : int, default=3
        Number of folds for cross-validation.

    problem_type : str, default='Regression'
        Either 'Classification' or 'Regression'.

    plot_metrics : bool, default=False
        Whether to generate model evaluation dashboard plots (only for classification).

    binary : bool, default=True
        Whether classification is binary (for ROC/PR curves).

    Returns:
    --------
    regression:
        float: overall RMSE score
        np.ndarray: average test predictions over CV folds

    classification:
        best_model : fitted model
        float: training accuracy
        float: test accuracy
        pd.DataFrame: metrics summary (Accuracy, Precision, Recall, F1)
    
    Notes:
    ------
    - For regression, it performs K-Fold CV manually and averages predictions.
    - For classification, it performs hyperparameter tuning with GridSearchCV and evaluates the best model.
    - The function automatically plots a Plotly dashboard (confusion matrix, ROC, learning curve, etc.).
    """

    if problem_type == 'Regression':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        predictions = np.zeros(len(X_train))
        test_predictions = np.zeros(len(X_test))
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            model = model_class(**params)
            model.fit(X_train[train_idx], y_train.iloc[train_idx])
            predictions[val_idx] = model.predict(X_train[val_idx])
            test_predictions += model.predict(X_test) / n_splits
        overall_rmse = np.sqrt(mean_squared_error(y_train, predictions))
        return overall_rmse, test_predictions

    elif problem_type == 'Classification':
        # --- Apply Grid Search ---
        model = GridSearchCV(model_class(), param_grid=params, cv=n_splits, n_jobs=-1, scoring='accuracy', verbose=1)

        # --- Fitting Model ---
        fit_model = model.fit(X_train, y_train)

        # --- Best Estimators ---
        best_model = model.best_estimator_
        print(f"\n{cf.clr.orange}Best Model:{cf.clr.reset} {best_model}")

        # --- Prediction Results ---
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # --- Accuracy & Best Score ---
        best_score = model.best_score_
        acc_score_train = accuracy_score(y_train, y_pred_train)
        acc_score_test = accuracy_score(y_test, y_pred_test) if y_test is not None else None

        print(f'{cf.clr.orange}Train Accuracy:{cf.clr.reset} {acc_score_train * 100:.2f}%')
        if y_test is not None:
            print(f'{cf.clr.orange}Test Accuracy:{cf.clr.reset} {acc_score_test * 100:.2f}%')
        
        print(f'{cf.clr.orange}Best Score:{cf.clr.reset} {best_score * 100:.2f}%')
        y_prob_test = model.predict_proba(X_test)[:, 1]


        # --- Visualization ---
        fig = viz.plot_model_evaluation(
            model=best_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            y_pred_test=y_pred_test,
            y_prob_test=y_prob_test,
            feature_names=X_train.columns.tolist(),
            show_feature_importance=show_feature_importance
        )
        fig.show()

        # metrics_results = met.metrics_calculator(y_pred_test, y_test, 'Classification', model_class.__name__)
        # return best_model, acc_score_train, acc_score, metrics_results
    




