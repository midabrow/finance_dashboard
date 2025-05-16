# utils/styles.py

STYLES = {
    "title": {
        "color": "#FFFFFF",
        "font_size": "28px",
        "font_weight": "bold",
        "margin_bottom": "1rem"
    },
    "subtitle": {
        "color": "#10B981",
        "font_size": "16px",
        "font_weight": "bold",
        "margin_top": "1.5rem",
        "margin_bottom": "0.5rem",
        "text_align": "center"
    }
}

def styled_text_with_const_style(text: str, style: dict) -> str:
    style_str = "; ".join([f"{k.replace('_', '-')}: {v}" for k, v in style.items()])
    return f"<div style='{style_str}'>{text}</div>"


def styled_text(
    text: str,
    color: str = "#333",
    font_size: str = "16px",
    font_weight: str = "bold",
    text_align: str = "center",
    margin_top: str = "1rem",
    margin_bottom: str = "0.5rem"
) -> str:
    return f"""
    <div style="
        color: {color};
        font-size: {font_size};
        font-weight: {font_weight};
        text-align: {text_align};
        margin-top: {margin_top};
        margin-bottom: {margin_bottom};
    ">
        {text}
    </div>
    """
