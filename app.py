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

    fig = px.scatter(
        df,
        x="strike",
        y="ask",
        title=f"{options_type.title()[:-1]} Options for {ticker_name}",
    )
    # fig.update_layout(xaxis_range=[left_bound_strike, right_bound_strike])
    fig.add_vline(x=current_stock_price, line_color="green")

    return fig

    # fig = px.scatter(
    #     df,
    #     x="strike",
    #     y="Bid",
    #     title=ticker,
    #     color="Expiry Date",
    #     hover_data=["Open Interest", "Implied Volatility", "Ask"],
    # ).update_traces(mode="lines+markers", marker_line_width=2, marker_size=8)

    # fig.add_vline(
    #     price,
    #     line_dash="dash",
    #     line_color="green",
    #     annotation_text=f"Current Stock Price: ${round(price, 2)}",
    # )
    # fig.update_layout(
    #     plot_bgcolor=colors["background"],
    #     paper_bgcolor=colors["background"],
    #     font_color=colors["text"],
    # )
    # return fig


ticker_name_dropdown = pn.widgets.Select(
    name="Ticker", value="QQQ", options=list(ALL_TICKERS["Symbol"])
)

expiry_dates_widget = pn.widgets.Select(
    name="Expiry Date",
    # size=5,
)

expiry_dates_widget.options = pn.bind(get_option_expiry_dates, ticker_name_dropdown)
expiry_dates_widget.value = expiry_dates_widget.options[0]

options_type_dropdown = pn.widgets.Select(
    name="Option Type", value="puts", options=["calls", "puts"]
)

number_of_strikes_dropdown = pn.widgets.Select(
    name="Number of Strikes to Show", value=20, options=list(range(10, 100, 10))
)

bound_plot = pn.bind(
    plot_options,
    ticker_name=ticker_name_dropdown,
    expiry_date=expiry_dates_widget,
    options_type=options_type_dropdown,
    number_of_strikes=number_of_strikes_dropdown,
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
    main=[bound_plot],
).servable()
