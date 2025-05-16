# ============================================================
# 🔹 IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.colors as colors
from plotly.subplots import make_subplots
from matplotlib import colors
import warnings
from ydata_profiling import ProfileReport
import missingno as msno
from termcolor import colored

import numpy as np
import matplotlib.pyplot as plt
from highlight_text import fig_text


















# ============================================================
# 🔹 KORELACJA
# ============================================================

def plot_correlation_heatmap(df, title_main, subtitle=None, author_tag=None, colorscale="RdBu", show_upper_triangle=False):
    """
    Create a Plotly-based correlation heatmap with optional title and annotations.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with numerical features.
    title_main : str
        Main title to display above the heatmap.
    subtitle : str or None
        Optional subtitle to display.
    author_tag : str or None
        Optional tag to display in the bottom right.
    colorscale : str
        Color scale for the heatmap.
    show_upper_triangle : bool
        Whether to show full matrix or mask the upper triangle.
    """
    corr = df.corr().round(2)

    if not show_upper_triangle:
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        corr = corr.mask(mask)

    heatmap = go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=colorscale,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation"),
        text=corr.values,
        texttemplate="%{text}",
        hovertemplate="Correlation between %{x} and %{y}: %{z}",
    )

    fig = go.Figure(data=[heatmap])

    fig.update_layout(
        title={
            "text": f"<b>{title_main}</b><br><span style='font-size:12px'>{subtitle if subtitle else ''}</span>",
            "x": 0.01, "xanchor": "left",
            "y": 0.95, "yanchor": "top"
        },
        margin=dict(l=60, r=60, t=80, b=40),
        xaxis=dict(tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        annotations=[
            dict(
                text=author_tag,
                x=1,
                y=-0.15,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=10, style="italic"),
                xanchor='right'
            )
        ] if author_tag else []
    )

    return fig







