from datetime import datetime
from functools import partial
from typing import Literal

import pandas as pd
import panel as pn
import plotly.express as px
import pandas as pd
import yfinance as yf


import numpy as np

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


STOCKER_TICKERS_CSV = "all_tickers.csv"

# Downloaded from https://github.com/paulperry/quant/blob/master/ETFs.csv
# Modified QQQ from QQQQ
ETF_TICKER_CSV = "ETFs.csv"


# TODO: This list of tickers needs updating!
def get_ticker_names(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


STOCK_TICKERS_DF = get_ticker_names(STOCKER_TICKERS_CSV)
ETF_TICKERS_DF = get_ticker_names(ETF_TICKER_CSV)

ALL_TICKERS = STOCK_TICKERS_DF.merge(
    ETF_TICKERS_DF, on=["Symbol", "Name", "Sector"], how="outer"
)


def get_option_expiry_dates(ticker_name: str) -> list[str]:
    return list(yf.Ticker(ticker_name).options)


def plot_options(
    ticker_name: str,
    expiry_date: str,
    options_type: Literal["calls", "puts"],
    number_of_strikes: int,
):
    ticker = yf.Ticker(ticker_name)

    current_stock_price = ticker.info["regularMarketPrice"]

    df: pd.DataFrame = getattr(ticker.option_chain(expiry_date), options_type)

    closest_strike_to_current_price = min(
        df["strike"], key=lambda strike: abs(strike - current_stock_price)
    )

    middle_index = df[df["strike"] == closest_strike_to_current_price].index
    df = df.iloc[
        int(middle_index.values[0])
        - int(number_of_strikes) // 2 : int(middle_index.values[0])
        + int(number_of_strikes) // 2
    ]

    # Think about pivoting the data so we can visualize bid ask spreads and volume all in one chart!
    fig = px.scatter(
        df,
        x="strike",
        y="ask",
        title=f"{options_type.title()[:-1]} Options for {ticker_name}",
    )
    # fig.update_layout(xaxis_range=[left_bound_strike, right_bound_strike])
    fig.add_vline(
        x=current_stock_price, line_color="green", annotation_text=current_stock_price
    )

    return fig


def plot_options_same_strike_across_expiries(
    ticker_name: str,
    strike_prices: list[str],
    options_type: Literal["calls", "puts"],
):
    ticker = yf.Ticker(ticker_name)
    ticker.option_chain()

    all_dfs = []
    expiries = list(ticker._expirations.keys())
    for expiry in expiries:
        # ONLY CALLS so far
        all_dfs.append(getattr(ticker.option_chain(expiry), options_type))
    all_dfs = pd.concat(all_dfs)

    all_dfs["expiry_date"] = (
        all_dfs["contractSymbol"]
        .str.extract(f"{ticker_name}(\\d+)")
        .apply(partial(pd.to_datetime, format="%y%m%d"))
    )
    sub_df = all_dfs[all_dfs["strike"].isin(strike_prices)]
    sub_df["strike"] = sub_df["strike"].astype("category")

    fig = px.scatter(
        sub_df,
        x="expiry_date",
        y="ask",
        color="strike",
        title=f"{options_type.title()[:-1]} Options for {ticker_name}",
        trendline="lowess",
        trendline_scope="trace",
    )
    return fig


def plot_stock(
    ticker_name: str,
):
    df = yf.download(
        ticker_name, start=str(datetime.today().strftime("%Y-%m-%d")), interval="1m"
    ).reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.reset_index(inplace=True)

    # Ensure the 'Datetime' column is in datetime format
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values(by="Datetime")

    fig = px.scatter(
        df,
        x="Datetime",
        y="Close",
        title=f"Stock Price for {ticker_name}",
    )

    return fig


ticker_name_dropdown = pn.widgets.Select(
    name="Ticker", value="TQQQ", options=list(ALL_TICKERS["Symbol"])
)

expiry_dates_widget = pn.widgets.Select(
    name="Expiry Date",
    # size=5,
)

expiry_dates_widget.options = pn.bind(get_option_expiry_dates, ticker_name_dropdown)
expiry_dates_widget.value = expiry_dates_widget.options[0]

options_type_dropdown = pn.widgets.Select(
    name="Option Type", value="calls", options=["calls", "puts"]
)

number_of_strikes_dropdown = pn.widgets.Select(
    name="Number of Strikes to Show", value=20, options=list(range(10, 100, 10))
)

bound_options_plot = pn.bind(
    plot_options,
    ticker_name=ticker_name_dropdown,
    expiry_date=expiry_dates_widget,
    options_type=options_type_dropdown,
    number_of_strikes=number_of_strikes_dropdown,
)

bound_options_by_expiry = pn.bind(
    plot_options_same_strike_across_expiries,
    ticker_name=ticker_name_dropdown,
    strike_prices=[85, 90, 95],
    options_type=options_type_dropdown,
)

bound_stock_plot = pn.bind(
    plot_stock,
    ticker_name=ticker_name_dropdown,
)

pn.template.MaterialTemplate(
    site="Options Dashboard",
    title="Single Expiry View",
    sidebar=[
        ticker_name_dropdown,
        expiry_dates_widget,
        options_type_dropdown,
        number_of_strikes_dropdown,
    ],
    main=[
        bound_stock_plot,
        bound_options_plot,
        bound_options_by_expiry,
    ],
).servable()
