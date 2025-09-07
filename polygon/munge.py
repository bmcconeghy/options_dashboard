import re
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# "Loose" because Polygon data is sometimes malformed
OCC_OPTION_NAME_PATTERN_LOOSE = re.compile(r"^O:([A-Z]+)(\d{6,7})([CP])(\d{6,9})$")


def clean_and_parse_option_names(
    df: pd.DataFrame, column_name: str = "ticker"
) -> pd.DataFrame:
    """Clean and parse Options Clearing Corporation (OCC) style option names.

    Args:
        df: Input DataFrame containing `column_name`.
        column: Name of the column containing option symbols.

    Returns:
        pd.DataFrame: DataFrame with additional parsed columns:
            ['symbol', 'expiry_date', 'option_type', 'strike_price', 'expiration']
    """
    result = df.copy()

    # Extract fields with lenient pattern to allow malformed but recoverable entries
    result[["symbol", "expiry_date", "option_type", "strike_price"]] = result[
        column_name
    ].str.extract(OCC_OPTION_NAME_PATTERN_LOOSE)

    # Drop rows that couldn't be parsed
    result.dropna(
        subset=["symbol", "expiry_date", "option_type", "strike_price"], inplace=True
    )
    if dropped_rows := len(df) - len(result):
        logger.info(f"Dropped {dropped_rows} rows with unparseable option names")

    # Normalize expiry_date: take last 6 digits (handles 7-digit errors like 2251219 -> 251219)
    result["expiry_date"] = result["expiry_date"].str[-6:]

    # Normalize strike_price: pad to 8 digits, truncate if longer
    result["strike_price"] = result["strike_price"].str.zfill(8).str[:8]
    result["strike_price"] = result["strike_price"].astype(float) / 1000

    result["expiry_date"] = pd.to_datetime(
        result["expiry_date"], format="%y%m%d", errors="coerce"
    )

    result.dropna(subset=["expiry_date"], inplace=True)
    if dropped_rows_due_to_date := len(df) - len(result):
        logger.info(
            f"Dropped {dropped_rows_due_to_date - dropped_rows} rows with unparseable option dates"
        )

    return result

def calculate_winners_and_losers(
    df: pd.DataFrame,
    stocks_of_interest: list[str] | None = None,
    num_winners: int = 10,
    num_losers: int = 10,
    ticker_column: str = "ticker",
) -> pd.DataFrame:
    """Calculate the percent gain or loss for all tickers (default) or stocks of interest.
    Number of stocks returned can also be specified for each type (i.e. Winner and Loser).
    """
    if not stocks_of_interest:
        stocks_of_interest = list(df[ticker_column].unique())
    only_stocks_of_interest = df[df[ticker_column].isin(stocks_of_interest)]
    assert (
        len(stocks_of_interest) == only_stocks_of_interest[ticker_column].nunique()
    ), f"There is a stock from stocks of interest missing in the input dataframe. Namely: {set(stocks_of_interest) - set(df['ticker'].unique())}"
    price_change = only_stocks_of_interest.groupby(ticker_column).agg(
        start_price=("close", "first"), end_price=("close", "last")
    )

    # only_stocks_of_interest.to_csv('Output.csv', index =True)
    # Calculate price change for each ticker
    price_change["percent_change"] = (
        (
            (price_change["end_price"] - price_change["start_price"])
            / price_change["start_price"]
        )
        * 100
    ).round(2)

    price_change["type"] = price_change.apply(
        lambda x: "Winner" if x["percent_change"] >= 0 else "Loser", axis=1
    )
    winners = (
        price_change[price_change["type"] == "Winner"]
        .sort_values(by="percent_change", ascending=False)
        .head(num_winners)
        .reset_index()
    )
    losers = (
        price_change[price_change["type"] == "Loser"]
        .sort_values(by="percent_change", ascending=True)
        .head(num_losers)
        .reset_index()
    )
    return pd.concat([winners, losers], ignore_index=True)