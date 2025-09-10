import logging

import polars as pl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def clean_and_parse_option_names(
    df: pl.DataFrame, column_name: str = "ticker"
) -> pl.DataFrame:
    """Clean and parse Options Clearing Corporation (OCC) style option names.

    Args:
        df: Input polars DataFrame containing `column_name`.
        column_name: Name of the column containing option symbols.

    Returns:
        pl.DataFrame: DataFrame with additional parsed columns:
            ['symbol', 'expiry_date', 'option_type', 'strike_price', 'expiration']
    """

    # "Loose" because Polygon data is sometimes malformed
    occ_option_name_pattern_loose = r"^O:([A-Z]+)(\d{6,7})([CP])(\d{6,9})$"
    extracted = df.with_columns(
        [
            pl.col(column_name)
            .str.extract(occ_option_name_pattern_loose, group_index=1)
            .alias("symbol"),
            pl.col(column_name)
            .str.extract(occ_option_name_pattern_loose, group_index=2)
            .alias("expiry_date"),
            pl.col(column_name)
            .str.extract(occ_option_name_pattern_loose, group_index=3)
            .alias("option_type"),
            pl.col(column_name)
            .str.extract(occ_option_name_pattern_loose, group_index=4)
            .alias("strike_price"),
        ]
    )

    # Drop rows that couldn't be parsed
    parsed = extracted.drop_nulls(
        subset=["symbol", "expiry_date", "option_type", "strike_price"]
    )
    dropped_rows = df.height - parsed.height
    if dropped_rows:
        logger.info(f"Dropped {dropped_rows} rows with unparseable option names")

    # Normalize expiry_date: take last 6 digits
    parsed = parsed.with_columns(
        pl.col("expiry_date").str.slice(-6, 6).alias("expiry_date")
    )

    # Normalize strike_price: pad to 8 digits, truncate if longer, then convert
    parsed = parsed.with_columns(
        (
            pl.col("strike_price").str.zfill(8).str.slice(0, 8).cast(pl.Float64) / 1000
        ).alias("strike_price")
    )

    # Parse expiry_date to datetime
    parsed = parsed.with_columns(
        pl.col("expiry_date").str.strptime(pl.Date, "%y%m%d", strict=False)
    )

    # Drop rows with unparseable dates
    final = parsed.drop_nulls(subset=["expiry_date"])
    dropped_rows_due_to_date = parsed.height - final.height
    if dropped_rows_due_to_date:
        logger.info(
            f"Dropped {dropped_rows_due_to_date} rows with unparseable option dates"
        )

    return final


def calculate_winners_and_losers(
    df: pl.DataFrame,
    stocks_of_interest: list[str] | None = None,
    num_winners: int = 10,
    num_losers: int = 10,
    ticker_column: str = "ticker",
) -> pl.DataFrame:
    """Calculate the percent gain or loss for all tickers (default) or stocks of interest.
    Number of stocks returned can also be specified for each type (i.e. Winner and Loser).
    """
    if not stocks_of_interest:
        stocks_of_interest = df[ticker_column].unique().to_list()
    only_stocks_of_interest = df.filter(pl.col(ticker_column).is_in(stocks_of_interest))
    missing_stocks = set(stocks_of_interest) - set(
        only_stocks_of_interest[ticker_column].unique().to_list()
    )
    assert len(missing_stocks) == 0, (
        f"There is a stock from stocks of interest missing in the input dataframe. Namely: {missing_stocks}"
    )

    price_change = (
        only_stocks_of_interest.group_by(ticker_column)
        .agg(
            [
                pl.col("close").first().alias("start_price"),
                pl.col("close").last().alias("end_price"),
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("end_price") - pl.col("start_price"))
                    / pl.col("start_price")
                    * 100
                )
                .round(2)
                .alias("percent_change"),
                pl.when(pl.col("end_price") - pl.col("start_price") >= 0)
                .then(pl.lit("Winner"))
                .otherwise(pl.lit("Loser"))
                .alias("type"),
            ]
        )
    )

    winners = (
        price_change.filter(pl.col("type") == "Winner")
        .sort("percent_change", descending=True)
        .head(num_winners)
    )
    losers = (
        price_change.filter(pl.col("type") == "Loser")
        .sort("percent_change", descending=False)
        .head(num_losers)
    )
    return pl.concat([winners, losers])
