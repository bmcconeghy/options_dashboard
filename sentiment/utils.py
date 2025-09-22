import re


def clean_for_finbert(text: str) -> str:
    """
    Cleans financial text for FinBERT input.
    Keeps relevant sentiment-bearing content.
    """

    # Replace smart punctuation with plain equivalents
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", '"')
        .replace("’", '"')
        .replace("–", "-")
        .replace("—", "-")
    )

    # Remove ticker symbols like (AAPL.O), ($MSFT), $TSLA
    text = re.sub(r"\(\$?[A-Z]{1,5}(\.[A-Z]{1,5})?\)", "", text)
    text = re.sub(r"\$[A-Z]{1,5}", "", text)

    # Remove known source names (Reuters, Bloomberg, etc.)
    text = re.sub(
        r"\b(Reuters|Bloomberg|CNBC|MarketWatch|WSJ|FT|Forbes)\b", "", text, flags=re.I
    )

    # Remove "Reporting by..." or "Editing by..." boilerplate
    text = re.sub(r"(Reporting|Editing|Edited) by.*", "", text, flags=re.I)

    # Normalize spacing
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)  # remove space before punctuation
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    text = text.strip()

    return text
