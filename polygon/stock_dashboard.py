import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import structlog
from dash import Input, Output, callback, dash_table, dcc, html
from munge import calculate_winners_and_losers, clean_and_parse_option_names
from scipy.interpolate import griddata

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

pd.options.plotting.backend = "plotly"
pd.set_option("display.max_columns", None)

logger = structlog.get_logger()


COLOURS = {"background": "#000000", "text": "#7FDBFF"}
ROOT_DIR = Path(os.environ.get("ROOT_DIR"))
STOCK_DIR = ROOT_DIR / "polygon/stock_csvs"
OPTION_DIR = ROOT_DIR / "polygon/options_csvs"


# Combine all stock data
combined_stock_df = pd.concat(
    [pd.read_csv(file) for file in STOCK_DIR.glob("*.csv.gz")],
    ignore_index=True,
)
combined_stock_df["window_start"] = (
    pd.to_datetime(combined_stock_df["window_start"])
    .dt.tz_localize("UTC")
    .dt.tz_convert("America/New_York")
)

# Combine all option data
combined_option_df = pd.concat(
    [pd.read_csv(file) for file in OPTION_DIR.glob("*.csv.gz")],
    ignore_index=True,
)
combined_option_df["window_start"] = (
    pd.to_datetime(combined_option_df["window_start"])
    .dt.tz_localize("UTC")
    .dt.tz_convert("America/New_York")
)

# Polygon has shitty looking data sometimes in terms of ticker structure, so we need to clean it up
combined_option_df = clean_and_parse_option_names(combined_option_df)

# Go through the combined list of stocks to find the ones with the biggest weekly gains and losses.
stock_watch = [
    "ALAB",
    "AVAV",
    "COIN",
    "CRCL",
    "LLY",
    "MDB",
    "NVDA",
    "PLTR",
    "TQQQ",
    "TSLA",
]

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
                        id="stock_name_dropdown",
                        options=[
                            {"label": i, "value": i}
                            for i in stock_and_option["ticker_stock"].unique()
                        ],
                        placeholder="Select Stock Ticker...",
                        value="NVDA",
                    ),
                    width=3,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="expiry_date_dropdown",
                        options=[
                            {"label": i, "value": i}
                            for i in stock_and_option["expiry_date"].unique()
                        ],
                        placeholder="Select Option Date...",
                    ),
                    width=3,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="option_type_dropdown",
                        options=[
                            {"label": "Put", "value": "P"},
                            {"label": "Call", "value": "C"},
                        ],
                        placeholder="Select Option Type...",
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


@app.callback(
    Output("expiry_date_dropdown", "options"),
    Input("stock_name_dropdown", "value"),
    prevent_initial_call=True,
)
def update_expiry_date_dropdown(value):
    """Not all stocks have the same expiry dates, so we need to filter the expiry date dropdown based on the selected stock."""
    return stock_and_option[stock_and_option["ticker_stock"] == value][
        "expiry_date"
    ].unique()


@callback(
    Output(
        component_id="my-output", component_property="children", allow_duplicate=True
    ),
    Input(component_id="stock_name_dropdown", component_property="value"),
    prevent_initial_call=True,
)
def update_output_div(input_value):
    return f"Output: {input_value}"


@callback(
    Output(component_id="histogram_graph", component_property="figure"),
    Input(component_id="stock_name_dropdown", component_property="value"),
)
def update_hist_graph(input_value):
    stock = combined_stock_df[combined_stock_df["ticker"] == f"{input_value}"]
    stock.loc[:, "Log_return"] = np.log(stock.loc[:, "close"] / stock.loc[:, "open"])
    fig = px.histogram(stock["Log_return"], range_x=[-0.01, 0.01])
    return fig


@callback(
    Output(component_id="option_topography", component_property="figure"),
    Input(component_id="stock_name_dropdown", component_property="value"),
    Input(component_id="expiry_date_dropdown", component_property="value"),
    Input(component_id="option_type_dropdown", component_property="value"),
    prevent_initial_call=True,
)
def update_option_graph(input_value1, input_value2, input_value3):
    logger.debug(f"{input_value1=}")
    logger.debug(f"{input_value2=}")
    logger.debug(f"{input_value3=}")
    # TODO: Is a scatter 3d plot better here?
    # TODO: Smooth out NaNs?

    stock_and_option_for_specific_stock = stock_and_option.query(
        "ticker_stock == @input_value1 & expiry_date == @input_value2 & option_type == @input_value3"
    )

    stock_and_option_for_specific_stock = stock_and_option_for_specific_stock.copy()
    stock_and_option_for_specific_stock["premium_to_stock_ratio"] = (
        stock_and_option_for_specific_stock["open_option"]
        / stock_and_option_for_specific_stock["open_stock"]
    )

    # Percent change by strike price
    stock_and_option_for_specific_stock["percent_change"] = (
        stock_and_option_for_specific_stock.groupby("strike_price")[
            "premium_to_stock_ratio"
        ].pct_change()
        * 100
    )
    pivot = stock_and_option_for_specific_stock.pivot(
        index="strike_price", columns="window_start", values="percent_change"
    )
    pivot = pivot.sort_index().sort_index(axis=1)

    # NGL: Got AI to write this interpolation method
    # Interpolate NaNs
    pivot_interp = pivot.copy()
    # Interpolate across time
    pivot_interp = pivot_interp.interpolate(axis=1, method="linear")
    # Interpolate across strike
    pivot_interp = pivot_interp.interpolate(axis=0, method="linear")
    # Edge fill
    pivot_interp = pivot_interp.fillna(method="bfill", axis=1).fillna(
        method="ffill", axis=1
    )

    fig = go.Figure(
        data=[
            go.Surface(
                z=pivot_interp.values,  # Interpolated % changes
                x=pivot_interp.columns,  # Timestamps
                y=pivot_interp.index.tolist(),  # Strike prices
                colorscale="Viridis",
                colorbar=dict(title="% Δ Ratio"),
                hovertemplate="Time: %{x}<br>Strike: %{y}<br>% Δ: %{z:.2f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="% Change in Option Premium / Stock Price Ratio",
        scene=dict(
            xaxis_title="Time",
            yaxis_title="Strike Price",
            zaxis_title="% Change in Ratio",
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=50),
    )
    return fig


@callback(
    Output(component_id="range_graph", component_property="figure"),
    Input(component_id="stock_name_dropdown", component_property="value"),
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
