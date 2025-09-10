import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import structlog
from dash import Input, Output, callback, dash_table, dcc, html
from munge import calculate_winners_and_losers, clean_and_parse_option_names

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

logger = structlog.get_logger()


COLOURS = {"background": "#000000", "text": "#7FDBFF"}
ROOT_DIR = Path(os.environ.get("ROOT_DIR"))
STOCK_DIR = ROOT_DIR / "polygon/stock_csvs"
OPTION_DIR = ROOT_DIR / "polygon/options_csvs"


# Combine all stock data
combined_stock_df = pl.concat(
    [pl.read_csv(file) for file in STOCK_DIR.glob("*.csv.gz")]
)
combined_stock_df = combined_stock_df.with_columns(
    pl.col("window_start")
    .cast(pl.Datetime("ns"))
    .dt.replace_time_zone("UTC")
    .dt.convert_time_zone("America/New_York")
)
# Combine all option data
combined_option_df = pl.concat(
    [pl.read_csv(file) for file in OPTION_DIR.glob("*.csv.gz")]
)
combined_option_df = combined_option_df.with_columns(
    pl.col("window_start")
    .cast(pl.Datetime("ns"))
    .dt.replace_time_zone("UTC")
    .dt.convert_time_zone("America/New_York")
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

stock_and_option = (
    combined_option_df.filter(pl.col("symbol").is_in(stock_watch))
    .sort(["window_start", "symbol"])
    .join_asof(
        combined_stock_df.filter(pl.col("ticker").is_in(stock_watch)).sort(
            ["window_start", "ticker"]
        ),
        left_on="window_start",
        right_on="window_start",
        by_left="symbol",
        by_right="ticker",
        suffix="_stock",
        strategy="nearest",
    )
)
winners_and_losers = calculate_winners_and_losers(combined_stock_df, stock_watch)
winners = winners_and_losers.filter(pl.col("type") == "Winner")
losers = winners_and_losers.filter(pl.col("type") == "Loser")


# App layout will have the drop down list and all calls for graphs
# Generate the drop down list of all unique ticker names and update all graphs when a new ticker is chosen

app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            [
                dbc.Col(
                    html.H1(
                        "Stock Exploration Dashboard",
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
                            for i in stock_and_option.select("symbol")
                            .unique()
                            .to_series()
                            .to_list()
                        ],
                        placeholder="Select Stock Ticker...",
                        value="NVDA",
                    ),
                    width=2,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="expiry_date_dropdown",
                        placeholder="Select Option Date...",
                    ),
                    width=2,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="option_type_dropdown",
                        options=[
                            {"label": "Put", "value": "P"},
                            {"label": "Call", "value": "C"},
                        ],
                        placeholder="Select Option Type...",
                        value="P",
                    ),
                    width=2,
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
                        data=winners.head(20).to_dicts(),
                        columns=[{"name": i, "id": i} for i in winners.columns],
                        page_size=20,
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left"},
                    ),
                    width=6,
                ),
                dbc.Col(
                    dash_table.DataTable(
                        data=losers.head(20).to_dicts(),
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
    Output("expiry_date_dropdown", "value"),  # Allows setting a default value
    Input("stock_name_dropdown", "value"),
)
def update_expiry_date_dropdown(value):
    """Not all stocks have the same expiry dates, so we need to filter the expiry date dropdown based on the selected stock."""
    unique_expiry_dates = sorted(
        stock_and_option.filter(pl.col("symbol") == value)
        .select("expiry_date")
        .unique()
        .to_series()
        .to_list()
    )
    return [
        {"label": item, "value": item} for item in unique_expiry_dates
    ], unique_expiry_dates[0]  # Set the first date as the default value


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
    stock = combined_stock_df.filter(pl.col("ticker") == input_value)
    stock = stock.with_columns(
        (pl.col("close") / pl.col("open")).log().alias("Log_return")
    ).to_pandas()
    fig = px.histogram(stock, x="Log_return", range_x=[-0.01, 0.01])
    fig.update_layout(
        title="Histogram of Daily Log Returns",
        xaxis_title="Log Return",
        yaxis_title="Count",
        bargap=0.1,
        plot_bgcolor=COLOURS["background"],
        paper_bgcolor=COLOURS["background"],
        font_color=COLOURS["text"],
    )
    return fig


@callback(
    Output(component_id="option_topography", component_property="figure"),
    Input(component_id="stock_name_dropdown", component_property="value"),
    Input(component_id="expiry_date_dropdown", component_property="value"),
    Input(component_id="option_type_dropdown", component_property="value"),
)
def update_option_graph(stock_name, expiry_date, option_type):
    # TODO: Is a scatter 3d plot better here?

    stock_and_option_for_specific_stock = stock_and_option.filter(
        (pl.col("symbol") == stock_name)
        & (pl.col("expiry_date").cast(str) == expiry_date)
        & (pl.col("option_type") == option_type)
    ).to_pandas()

    stock_and_option_for_specific_stock = stock_and_option_for_specific_stock.copy()
    stock_and_option_for_specific_stock["premium_to_stock_ratio"] = (
        stock_and_option_for_specific_stock["open"]
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
        ],
    )
    fig.update_layout(
        plot_bgcolor=COLOURS["background"],
        paper_bgcolor=COLOURS["background"],
        font_color=COLOURS["text"],
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
    stock = combined_stock_df.filter(pl.col("ticker") == input_value)
    stock = stock.with_columns(
        (pl.col("close") / pl.col("open")).log().alias("Log_return")
    )
    mean = stock["Log_return"].mean()
    vol = stock["Log_return"].std()
    mean_daily_effective = np.exp(mean) - 1
    vol_daily_effective = np.exp(vol) - 1
    first_price = stock["close"][0]
    n = stock.height
    high = [
        first_price
        * ((1 + mean_daily_effective) ** i)
        * ((1 + vol_daily_effective) ** np.sqrt(i))
        for i in range(n)
    ]
    low = [
        first_price
        * ((1 + mean_daily_effective) ** i)
        / ((1 + vol_daily_effective) ** np.sqrt(i))
        for i in range(n)
    ]
    df = stock.with_columns(
        [pl.Series("high", high), pl.Series("low", low)]
    ).to_pandas()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["window_start"].to_list(),
            y=df["close"].to_list(),
            mode="lines",
            name="Close",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["window_start"].to_list(),
            y=df["high"].to_list(),
            mode="lines",
            name="High",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["window_start"].to_list(),
            y=df["low"].to_list(),
            mode="lines",
            name="Low",
        )
    )
    fig.update_layout(
        title="Stock Price Range Projection",
        yaxis_type="log",
        xaxis_title="Date",
        yaxis_title="Price",
        plot_bgcolor=COLOURS["background"],
        paper_bgcolor=COLOURS["background"],
        font_color=COLOURS["text"],
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
