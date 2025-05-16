
from sklearn.metrics import confusion_matrix, roc_curve, auc
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import learning_curve
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


from helpers import model_utils as mdl

def plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive'], return_trace=False, show=False):
    cm = confusion_matrix(y_true, y_pred)

    heatmap = go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm.astype(str),
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    )

    if return_trace:
        return heatmap

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title="📊 Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        font=dict(size=12),
        width=500, height=500,
        margin=dict(t=50, b=50)
    )
    
    if show:
        fig.show()

    return fig  # zwróć pełen wykres


def plot_roc_curve(y_true, y_score, return_trace=False, show=False):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    
    roc_trace = go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}', line=dict(color='blue'))
    random_trace = go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='orange'), name='Random')

    if return_trace:
        return [roc_trace, random_trace]
    
    fig = go.Figure()
    fig.add_trace(roc_trace)
    fig.add_trace(random_trace)
    fig.update_layout(
        title="🚀 ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=450, height=400,
        font=dict(size=12)
    )
    if show:
        fig.show()

    return fig

    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5)
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

def plot_learning_curve(model, X_train, y_train, return_trace=False, show=False):
    """
    Generates a learning curve using Plotly. Works in standalone or subplot mode.

    Parameters:
    -----------
    model : estimator
        The machine learning model to evaluate.

    X_train : array-like
        Training features.

    y_train : array-like
        Training target values.

    return_trace : bool, default=False
        If True, returns Plotly traces for subplot usage.

    show : bool, default=False
        If True, shows the plot directly.

    Returns:
    --------
    - If return_trace=True: List of traces to be added to subplot.
    - Else: Plotly Figure object.
    """
    # --- Get data ---
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, n_jobs=-1)
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    # --- Prepare traces ---
    trace_train = go.Scatter(
        x=train_sizes,
        y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue')
    )
    trace_val = go.Scatter(
        x=train_sizes,
        y=test_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='orange')
    )

    # --- Return traces only (e.g. for subplot) ---
    if return_trace:
        return [trace_train, trace_val]

    # --- Full figure ---
    fig = go.Figure()
    fig.add_trace(trace_train)
    fig.add_trace(trace_val)

    fig.update_layout(
        title='📈 Learning Curve',
        xaxis_title='Training Samples',
        yaxis_title='Score',
        width=500,
        height=400,
        font=dict(size=12)
    )

    if show:
        fig.show()

    return fig


def plot_feature_importance_or_precision_recall(model, 
                                                feature_names=None, 
                                                y_test=None, 
                                                y_prob_test=None,
                                                topn=5,
                                                return_trace=False,
                                                show=False):
    """
    Plots either Top-N Feature Importances or Precision-Recall Curve depending on model capabilities.

    Parameters:
    -----------
    model : estimator
        Trained model (with or without feature_importances_).

    feature_names : list
        List of feature names (required for feature importances).

    y_test : array-like
        True labels (only required for Precision-Recall curve).

    y_prob_test : array-like
        Probabilities for positive class (only for Precision-Recall curve).

    topn : int
        Number of top features to display (default: 5).

    return_trace : bool
        If True, returns Plotly traces for subplot use.

    show : bool
        If True, displays the figure immediately.

    Returns:
    --------
    - traces (list) if return_trace == True
    - fig (go.Figure) otherwise
    """

    if hasattr(model, 'feature_importances_') and feature_names is not None:
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-topn:][::-1]
        top_features = [feature_names[i] for i in top_idx]
        top_values = importances[top_idx]

        trace = go.Bar(
            x=top_values[::-1],
            y=top_features[::-1],
            orientation='h',
            marker=dict(color='royalblue'),
            name='Feature Importance'
        )

        if return_trace:
            return [trace]

        fig = go.Figure()
        fig.add_trace(trace)
        fig.update_layout(
            title='🔥 Top Feature Importances',
            xaxis_title='Importance',
            yaxis_title='Feature',
            width=500, height=400,
            font=dict(size=12)
        )
        if show:
            fig.show()
        return fig

    else:
        if y_test is None or y_prob_test is None:
            raise ValueError("y_test and y_prob_test must be provided for Precision-Recall Curve.")

        precision, recall, _ = precision_recall_curve(y_test, y_prob_test)

        trace = go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name='Precision-Recall',
            line=dict(color='blue')
        )

        if return_trace:
            return [trace]

        fig = go.Figure()
        fig.add_trace(trace)
        fig.update_layout(
            title='🔁 Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=500, height=400,
            font=dict(size=12)
        )
        if show:
            fig.show()
        return fig


def plot_model_evaluation(model, X_train, y_train, X_test, y_test, y_pred_test, y_prob_test, feature_names, show_feature_importance=True):

    # === 0. Create Subplots ===
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Confusion Matrix",
            "ROC AUC Curve",
            "Learning Curve",
            "Feature Importances" if show_feature_importance is not None else "Precision-Recall Curve"
        )
    )

    # === 1. Confusion Matrix ===
    cm_trace = plot_confusion_matrix(y_test, y_pred_test, return_trace = True)
    fig.add_trace(cm_trace, row=1, col=1)
    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(showticklabels=True)

    # === 2. ROC Curve ===
    roc_traces = plot_roc_curve(y_test, y_pred_test, return_trace = True)
    fig.add_trace(roc_traces[0], row=1, col=2)
    fig.add_trace(roc_traces[1], row=1, col=2)


    # === 3. Learning Curve ===
    learn_traces = plot_learning_curve(model, X_train, y_train, return_trace=True)
    fig.add_trace(learn_traces[0], row=2, col=1)
    fig.add_trace(learn_traces[1], row=2, col=1)

    # === 4. Feature Importances or Precision-Recall Curve ===
    traces = plot_feature_importance_or_precision_recall(
        model, 
        feature_names=X_train.columns.tolist(), 
        y_test=y_test, 
        y_prob_test=y_prob_test,
        return_trace=True
    )
    for trace in traces:
        fig.add_trace(trace, row=2, col=2)

    # === 5. Update layout ===
    fig.update_layout(
        title=dict(
            text="📊 Model Evaluation Dashboard",
            x=0.5,
            xanchor='center',
            font=dict(
                size=25,
                family='Arial',
                color='black',
                weight='bold'
            )
        ),
        height=800,
        width=1000,
        showlegend=False,
        template='simple_white',

        # Axes formatting
        xaxis1=dict(title=dict(text='Predicted Class', font=dict(size=12, color='black', weight='bold'))),
        yaxis1=dict(title=dict(text='True Class', font=dict(size=12, color='black', weight='bold'))),
        xaxis2=dict(title=dict(text='False Positive Rate', font=dict(size=12, color='black', weight='bold'))),
        yaxis2=dict(title=dict(text='True Positive Rate', font=dict(size=12, color='black', weight='bold'))),
        xaxis3=dict(title=dict(text='Training Instances', font=dict(size=12, color='black', weight='bold'))),
        yaxis3=dict(title=dict(text='Scores', font=dict(size=12, color='black', weight='bold'))),
        xaxis4=dict(title=dict(text='Importance', font=dict(size=12, color='black', weight='bold'))),
        yaxis4=dict(title=dict(text='Feature', font=dict(size=12, color='black', weight='bold'))),
    )


    return fig



def plot_performance_summary(y_pred, y_test, model_class, ax, title_style):

    result = mdl.metrics_calculator(y_pred, y_test, model_class.__name__)

    # Tworzenie tabeli
    table = ax.table(cellText=result.values, colLabels=result.columns, rowLabels=result.index, loc='best')
    table.scale(0.6, 3.6)
    table.set_fontsize(12)

    # Ukrywanie osi dla obszaru tabeli
    ax.axis('tight')
    ax.axis('off')

    # Tytuł tabeli
    ax.set_title('Performance Summary on Test Data\n', **title_style)

    # Zmiana koloru wiersza nagłówka tabeli
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Nagłówek tabeli
            cell.set_color('#1e3a8a')
            cell.get_text().set_color('white')
