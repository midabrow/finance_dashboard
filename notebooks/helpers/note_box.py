# helpers/note_box.py

from IPython.display import HTML

def make_note_box(text: str, title: str = '🧠 Note', color: str = 'yellow') -> HTML:
    """
    Creates a styled HTML note box for Jupyter notebooks.

    Parameters:
    -----------
    text : str
        The content of the note.
    title : str
        Optional title shown at the top of the box.
    color : str
        Box color. Options: 'yellow', 'blue', 'red', 'green'.

    Returns:
    --------
    IPython.display.HTML
        Renderable HTML block.
    """
    color_styles = {
        'yellow': {
            'background': '#FFF9C4',
            'border': '#FBC02D',
            'text': '#555'
        },
        'blue': {
            'background': '#D0E7FF',
            'border': '#1E3A8A',
            'text': '#1E3A8A'
        },
        'red': {
            'background': '#FFCDD2',
            'border': '#C62828',
            'text': '#333'
        },
        'green': {
            'background': '#C8E6C9',
            'border': '#388E3C',
            'text': '#2E7D32'
        }
    }

    style = color_styles.get(color, color_styles['yellow'])

    html = f"""
    <div style="
        background-color: {style['background']};
        border-left: 6px solid {style['border']};
        padding: 14px;
        margin: 20px 0;
        border-radius: 8px;
        font-family: Arial, sans-serif;
        font-size: 15px;
        max-width: 900px;
        color: {style['text']};
        line-height: 1.6;
    ">
        <b>{title}:</b><br><br>
        {text}
    </div>
    """
    return HTML(html)

from IPython.display import HTML

def make_dataset_summary_box(rows: int, cols: int, dtypes: dict, missing_info: str) -> HTML:
    """
    Tworzy kafelek z podsumowaniem zbioru danych na podstawie dynamicznych typów danych.

    Parameters:
    - rows: liczba wierszy w zbiorze
    - cols: liczba kolumn w zbiorze
    - dtypes: słownik {typ: liczba kolumn}, np. {'int64': 10, 'float64': 2}
    - missing_info: string opisujący brakujące dane

    Returns:
    - IPython.display.HTML kafelek gotowy do wstawienia do notebooka
    """

    # Sekcja typów danych jako <li>
    dtypes_list = "".join([
        f"<li>{count} column{'s' if count > 1 else ''} of type <code>{dtype}</code></li>"
        for dtype, count in dtypes.items()
    ])

    html = f"""
    <div style="
        background-color: #D0E7FF;
        border-left: 5px solid #1E3A8A;
        padding: 16px;
        margin: 20px 0;
        border-radius: 8px;
        font-family: Arial, sans-serif;
        font-size: 15px;
        max-width: 900px;
        color: #1E3A8A;
        line-height: 1.6;
    ">
        <b>📋 Dataset Overview:</b><br><br>

        ✅ <b>Number of Entries:</b> The dataset contains <b>{rows}</b> rows and <b>{cols}</b> columns.<br>
        ✅ <b>Data Types:</b>
        <ul style="margin-top: 5px;">
            {dtypes_list}
        </ul>
        ✅ <b>Missing Values:</b> {missing_info}
    </div>
    """
    return HTML(html)
