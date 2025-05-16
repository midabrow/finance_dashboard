# ============================================================
# 🔹 IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from helpers import config as cf
from termcolor import colored
from matplotlib import colors
import matplotlib.ticker as ticker

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from ydata_profiling import ProfileReport
import missingno as msno

from scipy.stats import zscore
from scipy.stats import normaltest


# ============================================================
# 🔹 UNIQUE VALUES FUNCTIONS
# ============================================================

def check_unique_values(dataset, dataset_name):
    print(cf.clr.orange + f"\n{'='*50}")
    print(f"🔍 Unique Values Report for: {cf.clr.bold + dataset_name}")
    print(f"{'='*50}" + cf.clr.reset)

    unique_values = dataset.apply(lambda x: x.nunique())
    print("\nNumber of unique values per column:")
    print(unique_values)

    object_columns = dataset.select_dtypes(include=['object']).columns
    if len(object_columns) == 0:
        print(colored('❌ There is no object data types in datase', 'red'))
    else:
        for col in object_columns:
            print(f"\nUnique values in column '{col}':")
            print(dataset[col].unique())



# ============================================================
# 🔹 MISSING VALUES FUNCTIONS
# ============================================================
def plot_missing_values(dataset, dataset_name):
    nan_percent = dataset.isnull().mean() * 100

    # Filtering features with missing value
    nan_percent= nan_percent[nan_percent>0].sort_values()
    nan_percent = round(nan_percent,1)

    # Plot the barh chart
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(nan_percent.index, nan_percent.values, color='#1E3A8A', height=0.65)

    # Annotate the values and indexes
    for i, (value, name) in enumerate(zip(nan_percent.values, nan_percent.index)):
        ax.text(value+0.5, i, f"{value}%", ha='left', va='center', fontweight='bold', color='#1E3A8A', fontsize=12)
    
    # Set x-axis limit
    ax.set_xlim([0,110])

    # Add title and xlabel
    plt.title("Percentage of Missing Data in " + dataset_name, fontsize=14)
    plt.xlabel('Percentages (%)', fontsize=12)
    plt.show()


def report_missing_data(dataset, dataset_name):
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent = dataset.isnull().mean() * 100
    
    result = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    print(cf.clr.orange + f"🔴 Missing Data Report for: {dataset_name}")
    print(f"{'-'*50}" + cf.clr.reset)

    print(result[result['Percent'] > 0])

    if dataset.isnull().sum().sum() != 0:
        print(f"\n⚠️ Missing Data Matrix for: {dataset_name}")
        plt.figure(figsize=(10, 5))  
        msno.matrix(dataset)
        plt.title(f"Missing Data in {dataset_name}", fontsize=14) 
        plt.show()

        print(cf.clr.orange + f"\n{'='*50}")
        print(f"🔍 Report for: {cf.clr.bold + dataset_name}")
        print(f"{'='*50}" + cf.clr.reset)
        plot_missing_values(dataset, dataset_name)

def show_percent_of_missing_values(dateset):
    missing_vals = round(dateset.isna().mean() * 100, 1)
    print(cf.clr.orange + "Columns with missing values:" + cf.clr.reset)
    print(missing_vals[missing_vals > 0])
    return 

# ============================================================
# 🔹 NUMERICAL COLUMNS
# ============================================================


def show_dtypes_cols(dataFrame, dtype, datasetType='Train'):
    cols = dataFrame.select_dtypes(include=dtype).columns.to_list()
    print(cf.clr.orange + f'.: {datasetType} Dataset - Numerical Columns :.' + cf.clr.reset)
    print(cols)
    return colsfplot_correlation_heatmap


def plot_numerical_distributions_plotly(df, numerical_cols, title='Distribution of Numerical Variables'):
    """
    Generates interactive histograms for numerical columns using Plotly.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_cols (list): List of numerical column names.
        title (str): Title for the overall plot.
    
    Returns:
        plotly.graph_objs.Figure: The combined figure.
    """
    
    
    n_cols = 2
    n_rows = int(np.ceil(len(numerical_cols) / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=numerical_cols)

    for idx, col in enumerate(numerical_cols):
        row = (idx // n_cols) + 1
        col_pos = (idx % n_cols) + 1
        
        # Compute binning strategy
        if df[col].dtype == 'int64':
            bin_edges = np.arange(df[col].min(), df[col].max() + 2) - 0.5
        else:
            bin_edges = np.histogram_bin_edges(df[col].dropna(), bins='sturges')
        
        # Create histogram
        hist = go.Histogram(
            x=df[col],
            xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=np.diff(bin_edges).mean()),
            marker=dict(color='#1E3A8A', line=dict(color='white', width=1)),
            opacity=0.75,
            name=col,
            showlegend=False
        )
        
        # Add to subplot
        fig.add_trace(hist, row=row, col=col_pos)

        # Add annotation with mean and std
        mean = df[col].mean()
        std = df[col].std()
        annotation_text = f"μ={mean:.2f}<br>σ={std:.2f}"

        xref = f"x{idx+1}" if idx > 0 else "x"
        yref = f"y{idx+1}" if idx > 0 else "y"

        fig.add_annotation(
            xref=f"{xref} domain",
            yref=f"{yref} domain",
            x=0.85, y=0.95,
            text=annotation_text,
            showarrow=False,
            font=dict(size=11, color="white"),
            align="right",
            bgcolor="#1E3A8A",
            bordercolor="white",
            borderwidth=1,
            row=row, col=col_pos
        )
    
    fig.update_layout(
        height=300 * n_rows,
        title_text=title,
        title_font=dict(size=20, color='#1E3A8A'),
        template='plotly_white',
        margin=dict(t=50)
    )
    
    return fig



# ============================================================
# 🔹 CATEGORICAL COLUMNS
# ============================================================
def plot_categorical_pie(df: pd.DataFrame, categorical_columns: list):
    """
    Generates pie charts for categorical variables in a given DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    categorical_columns (list): List of categorical column names.
    """
    # Create subplots
    fig = make_subplots(rows=1, cols=len(categorical_columns), specs=[[{'type':'domain'}]*len(categorical_columns)],
                        vertical_spacing=0.01, horizontal_spacing=0.01)
    
    for i, feature in enumerate(categorical_columns):
        value_counts = df[feature].value_counts()
        labels = value_counts.index.tolist()
        values = value_counts.values.tolist()

        # Define color map based on purple
        cmap = colors.LinearSegmentedColormap.from_list("Purple", ["Purple", "white"])
        norm = colors.Normalize(vmin=0, vmax=len(labels))
        color_list = [colors.rgb2hex(cmap(norm(i))) for i in range(len(labels))]

        # Create the pie chart
        pie_chart = go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=color_list, line=dict(color='white', width=3)),
            textposition='inside',
            textinfo='percent+label',
            title=feature,
            title_font=dict(size=25, color='black', family='Calibri')
        )

        # Add the pie chart to the subplot
        fig.add_trace(pie_chart, row=1, col=i+1)

    # Update the layout
    fig.update_layout(showlegend=False, height=400, width=990, 
                      title={
                          'text': "Distribution of Categorical Variables",
                          'y': 0.90,
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top',
                          'font': {'size': 28, 'color': 'black', 'family': 'Calibri'}
                      })
    
    # Show the plot
    fig.show()


# ============================================================
# 🔹 DUPLICATE VALUES FUNCTIONS
# ============================================================
def check_duplicates(dataset, dataset_name):
    print(cf.clr.orange + f"\n{'='*50}")
    print(f"🔍 Report for: {cf.clr.bold + dataset_name}")
    print(f"{'='*50}" + cf.clr.reset)
    
    total_duplicates = dataset.duplicated().sum()  # Liczba zduplikowanych wierszy
    print(cf.clr.bold + cf.clr.blue + f"\nTotal number of duplicate rows: {total_duplicates}" + cf.clr.reset)
    
    if total_duplicates > 0:
        print("\nDuplicate rows in the dataset:")
        print(dataset[dataset.duplicated()])
    
    print(f"\nChecking duplicates in each column:")
    for col in dataset.columns:
        duplicate_values = dataset[col].duplicated().sum()
        print(f"Column '{cf.clr.bold + col + cf.clr.reset}' has {duplicate_values} duplicate values")
    
    print(f"{'='*50}\n")



# ============================================================
# 🔹 CLEANING DATA
# ============================================================
def clean_column_names(dataset: pd.DataFrame, dataset_name: str, to_lower: bool = True, 
                        replace_chars: bool = False, old_char: str = None, new_char: str = None, 
                        remove_special_chars: bool = False):
    """
    Cleans column names in a dataset by applying transformations such as:
    - Stripping leading/trailing spaces
    - Converting to lowercase (optional)
    - Replacing specified characters (optional)
    - Removing special characters (optional)
    - Checking for duplicate column names

    Parameters:
    - dataset (pd.DataFrame): The dataset whose column names need to be cleaned.
    - dataset_name (str): Name of the dataset (for display purposes).
    - to_lower (bool): If True, converts all column names to lowercase (default is True).
    - replace_chars (bool): If True, replaces occurrences of old_char with new_char in column names (default is False).
    - old_char (str): Character(s) to be replaced when replace_chars is True.
    - new_char (str): Character(s) to replace old_char when replace_chars is True.
    - remove_special_chars (bool): If True, removes non-alphanumeric characters from column names (default is True).

    Returns:
    - None (modifies dataset in place)
    """
    print(cf.clr.orange + f"\n{'='*50}")
    print(f"🔍 Checking and cleaning column names for: {cf.clr.bold}{dataset_name}")
    print(f"{'='*50}" + cf.clr.reset)
    
    original_columns = dataset.columns.tolist()
    print(f"\nOriginal column names: {original_columns}")
    
    cleaned_columns = []
    for col in original_columns:
        col = col.strip()  # Remove leading/trailing spaces
        
        if replace_chars and old_char and new_char:
            col = col.replace(old_char, new_char)  # Replace specified character
        
        if to_lower:
            col = col.lower()  # Convert to lowercase
        
        if remove_special_chars:
            col = ''.join(e if e.isalnum() or e == '_' else '' for e in col)  # Remove special characters
        
        cleaned_columns.append(col)
    
    dataset.columns = cleaned_columns
    print(f"\nCleaned column names: {dataset.columns.tolist()}")
    
    # Check for duplicate column names
    duplicates = dataset.columns[dataset.columns.duplicated()].tolist()
    if duplicates:
        print(f"\n⚠️ Duplicate columns found: {duplicates}")
    else:
        print("\n✅ No duplicate columns found.")
    

# ============================================================
# 🔹 TRUE NUMERICAL 
# ============================================================

def detect_semantic_categoricals(df, max_unique=10, exclude_target=None):
    """
    Detects numerical columns that semantically behave like categorical features.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataset.
    max_unique : int
        Maximum number of unique values for a column to be considered categorical.
    exclude_target : str or list
        Column(s) to exclude from the check (e.g., the target column).

    Returns:
    --------
    List[str]
        A list of column names that are likely to be semantic categoricals.
    """

    if exclude_target is None:
        exclude_target = []
    elif isinstance(exclude_target, str):
        exclude_target = [exclude_target]

    candidates = []
    for col in df.select_dtypes(include=["int", "float"]).columns:
        if col in exclude_target:
            continue
        unique_vals = df[col].nunique()
        if unique_vals <= max_unique:
            candidates.append(col)
    return candidates



# ============================================================
# 🔹 OUTLIERS
# ============================================================

def check_outliers(dataset, dataset_name, numerical_cols=None, target_col=None):
    """
    Interactive outlier report using Plotly Express, with normality test check.

    Parameters:
    -----------
    dataset : pd.DataFrame
        The dataset to check.
    dataset_name : str
        Name of the dataset.
    numerical_cols : list or None
        List of numerical columns to analyze. If None, all numerical columns are used.
    target_col : str or None
        Optional target column for coloring scatter plots.
    """
    print(f"\n{'='*60}")
    print(f"🔍 Outliers Report for: {dataset_name}")
    print(f"{'='*60}")

    if numerical_cols is None:
        numerical_cols = dataset.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        print(f"\n📊 Checking outliers in column: {col}")

        # IQR outliers
        Q1, Q3 = dataset[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers_iqr = dataset[(dataset[col] < lower_bound) | (dataset[col] > upper_bound)]
        print(f"📌 IQR Outliers (<= {lower_bound:.2f} or >= {upper_bound:.2f}): {len(outliers_iqr)}")


        # Test normalności
        try:
            stat, p = normaltest(dataset[col].dropna())
            print(f"🧪 Normality Test (D’Agostino & Pearson): stat={stat:.2f}, p-value={p:.4f}")
            if p < 0.05:
                print("⚠️ Data is not normally distributed. Z-score may not be appropriate.")
            else:
                print("✅ Data appears to be normally distributed. Z-score can be used.")
                
            # Z-score outliers
            if p >= 0.05:
                z_scores = zscore(dataset[col].fillna(0))
                outliers_z = dataset[np.abs(z_scores) > 3]
                print(f"📌 Z-score Outliers (>|3|): {len(outliers_z)}")
   
            else:
                print("ℹ️ Skipping Z-score outlier detection due to non-normality.")

        except Exception as e:
            print(f"⚠️ Normality test failed: {e}")

        # Plotly Boxplot
        fig_box = px.box(dataset, x=col, title=f"Boxplot of {col}", points="all", template="plotly_white")
        fig_box.update_traces(marker_color="#1E3A8A", line_color="black")
        fig_box.show()

        # Plotly Scatterplot
        if target_col:
            fig_scatter = px.scatter(
                dataset, x=col, y=target_col,
                color=target_col,
                title=f"Scatterplot of {col} vs {target_col}",
                template="plotly_white",
                color_continuous_scale="RdBu"
            )
            fig_scatter.update_traces(marker=dict(size=6, opacity=0.7, line=dict(width=0.5, color='white')))
            fig_scatter.show()