import logging
import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import callback, dcc, dash_table, html, Input, Output

from polygon.munge import clean_and_parse_option_names, calculate_winners_and_losers

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

pd.options.plotting.backend = "plotly"
pd.set_option("display.max_columns", None)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


COLOURS = {"background": "#000000", "text": "#7FDBFF"}
ROOT_DIR = Path(os.environ.get("ROOT_DIR"))
STOCK_DIR = ROOT_DIR / "polygon/stock_csvs"
OPTION_DIR = ROOT_DIR / "polygon/options_csvs"


# Combine all stock data
combined_stock_df = pd.concat(
    [pd.read_csv(file) for file in STOCK_DIR.glob("*.csv.gz")],
    ignore_index=True,
)
combined_stock_df["window_start"] = pd.to_datetime(combined_stock_df["window_start"])

# Combine all option data
combined_option_df = pd.concat(
    [pd.read_csv(file) for file in OPTION_DIR.glob("*.csv.gz")],
    ignore_index=True,
)
combined_option_df["window_start"] = pd.to_datetime(combined_option_df["window_start"])

# Polygon has shitty looking data sometimes in terms of ticker structure, so we need to clean it up
combined_option_df = clean_and_parse_option_names(combined_option_df)

# Go through the combined list of stocks to find the ones with the biggest weekly gains and losses.
stock_watch = ["ALAB", "AVAV", "COIN", "CRCL", "LLY", "MDB", "NVDA", "PLTR", "TQQQ", "TSLA"]

stock_and_option = pd.merge_asof(
    left=combined_option_df[combined_option_df["symbol"].isin(stock_watch)].sort_values(
    by=["window_start", "symbol"]
),
    right=combined_stock_df[combined_stock_df["ticker"].isin(stock_watch)].sort_values(
    by=["window_start", "ticker"]
),
    on=["window_start"],
    left_by=["symbol"],
    right_by=["ticker"],
    suffixes=("_option", "_stock"),
    direction="nearest",
)
winners_and_losers = calculate_winners_and_losers(combined_stock_df, stock_watch)
winners = winners_and_losers[winners_and_losers["type"] == "Winner"]
losers = winners_and_losers[winners_and_losers["type"] == "Loser"]


# App layout will have the drop down list and all calls for graphs
# Generate the drop down list of all unique ticker names and update all graphs when a new ticker is chosen

app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            [
                dbc.Col(
                    html.H1(
                        "Stock Selection",
                        className="text-center mb-4",
                        style={"color": COLOURS["text"]},
                    )
                )
            ]
        ),
        # Dropdown Row
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="stk_dropdown",
                        options=[
                            {"label": i, "value": i} for i in stock_and_option["ticker_stock"].unique()
                        ],
                        placeholder="Select Stock Ticker...",
                        value="NVDA",
                    ),
                    width=3,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="date_dropdown",
                        options=[
                            {"label": i, "value": i}
                            for i in stock_and_option["expiry_date"].unique()
                        ],
                        placeholder="Select Option Date...",
                    ),
                    width=3,
                ),
            ],
            className="mb-4",
        ),
        # Output Div
        dbc.Row([dbc.Col(html.Div(id="my-output"))]),
        # Graphs: Topography + Histogram
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="option_topography", style={"height": "600px"}),
                    width=8,
                ),
                dbc.Col(
                    dcc.Graph(id="histogram_graph", style={"height": "600px"}), width=4
                ),
            ],
            className="mb-4",
        ),
        # Full-Width Range Graph
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="range_graph", style={"height": "500px"}),
                )
            ],
            className="mb-4",
        ),
        # Winners and Losers Tables
        dbc.Row(
            [
                dbc.Col(
                    dash_table.DataTable(
                        data=winners.head(20).to_dict("records"),
                        columns=[{"name": i, "id": i} for i in winners.columns],
                        page_size=20,
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left"},
                    ),
                    width=6,
                ),
                dbc.Col(
                    dash_table.DataTable(
                        data=losers.head(20).to_dict("records"),
                        columns=[{"name": i, "id": i} for i in losers.columns],
                        page_size=20,
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left"},
                    ),
                    width=6,
                ),
            ],
            className="mb-5",
        ),
    ],
    fluid=True,
    style={"backgroundColor": COLOURS["background"], "padding": "20px"},
)


@callback(
    Output(
        component_id="my-output", component_property="children", allow_duplicate=True
    ),
    Input(component_id="stk_dropdown", component_property="value"),
    prevent_initial_call=True,
)
def update_output_div(input_value):
    return f"Output: {input_value}"


@callback(
    Output(component_id="histogram_graph", component_property="figure"),
    Input(component_id="stk_dropdown", component_property="value"),
)
def update_hist_graph(input_value):
    stock = combined_stock_df[combined_stock_df["ticker"] == f"{input_value}"]
    stock.loc[:, "Log_return"] = np.log(stock.loc[:, "close"] / stock.loc[:, "open"])
    fig = px.histogram(stock["Log_return"], range_x=[-0.01, 0.01])
    return fig


@callback(
    Output(component_id="option_topography", component_property="figure"),
    Input(component_id="stk_dropdown", component_property="value"),
    Input(component_id="date_dropdown", component_property="value"),
)
def update_option_graph(input_value1, input_value2):
    print(f"{input_value1=}")
    print(f"{input_value2=}")
    stock_opt = stock_and_option.query(
        "ticker_stock == @input_value1 & expiry_date == @input_value2"
    )
    y_data = stock_opt["strike_price"].unique().tolist()
    x_data = stock_opt["window_start"].unique().tolist()
    z = pd.DataFrame()
    for strike_price in y_data:
        z[strike_price] = stock_opt[stock_opt["strike_price"] == strike_price][
            "open_option"
        ].reset_index(drop=True)
    fig = go.Figure(data=[go.Surface(x=x_data, y=y_data, z=z.transpose())])
    return fig


@callback(
    Output(component_id="range_graph", component_property="figure"),
    Input(component_id="stk_dropdown", component_property="value"),
)
def update_range_graph(input_value):
    stock = combined_stock_df[combined_stock_df["ticker"] == f"{input_value}"]
    stock.loc[:, ("Log_return")] = np.log(
        stock.loc[:, ("close")] / stock.loc[:, ("open")]
    )
    mean = np.mean(stock.loc[:, "Log_return"])
    vol = np.std(stock.loc[:, "Log_return"])
    # mean_annual_log=252*mean
    # vol_annual_log=252**0.5*vol
    # mean_annual_effective=np.exp(mean_annual_log)-1
    # vol_annual_effective=np.exp(vol_annual_log)-1
    SAP = np.average(stock["close"])
    stock["price_dif"] = np.square(stock.loc[:, ("close")] - SAP)
    Dif_Sum = np.sum(stock.loc[:, ("price_dif")])
    Varience = Dif_Sum / (len(stock["price_dif"] - 1))
    SV = np.sqrt(Varience)
    mean_daily_effective = np.exp(mean) - 1
    vol_daily_effective = np.exp(vol) - 1
    first_price = stock.close.iloc[0]
    stock["high"] = first_price * np.array(
        [
            ((1 + mean_daily_effective) ** i)
            * ((1 + vol_daily_effective) ** np.sqrt(i))
            for i in range(len(stock))
        ]
    )
    stock["low"] = first_price * np.array(
        [
            ((1 + mean_daily_effective) ** i)
            / ((1 + vol_daily_effective) ** np.sqrt(i))
            for i in range(len(stock))
        ]
    )
    fig = stock.plot(x="window_start", y=["close", "high", "low"], log_y=True)
    return fig


if __name__ == "__main__":
    app.run(debug=True)
