import os
from pathlib import Path

import numpy as np
import panel as pn
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import structlog
from fetch import download_file_from_s3, get_newest_flat_files_for_prefix
from munge import calculate_winners_and_losers, clean_and_parse_option_names

pn.extension("plotly")
pn.extension("tabulator")


logger = structlog.get_logger()

COLOURS = {"background": "#000000", "text": "#7FDBFF"}
ROOT_DIR = Path(os.environ.get("ROOT_DIR"))
STOCK_DIR = ROOT_DIR / "polygon/stocks_csvs"
OPTION_DIR = ROOT_DIR / "polygon/options_csvs"

STOCK_DIR.mkdir(parents=True, exist_ok=True)
OPTION_DIR.mkdir(parents=True, exist_ok=True)

NUM_DAYS_TO_FETCH = 2

most_recent_stock_data = get_newest_flat_files_for_prefix(
    base_prefix="us_stocks_sip", level="minute_aggs_v1", num_files=NUM_DAYS_TO_FETCH
)
most_recent_option_data = get_newest_flat_files_for_prefix(
    base_prefix="us_options_opra", level="minute_aggs_v1", num_files=NUM_DAYS_TO_FETCH
)


combined_stock_dfs = []
for stock_data in most_recent_stock_data:
    local_path = STOCK_DIR / stock_data.split("/")[-1]
    if not local_path.exists():
        local_path = download_file_from_s3(
            object_key=stock_data,
            root_dir=STOCK_DIR,
        )
    combined_stock_dfs.append(pl.read_csv(local_path))

combined_option_dfs = []
for options_data in most_recent_option_data:
    local_path = OPTION_DIR / options_data.split("/")[-1]
    if not local_path.exists():
        download_file_from_s3(
            object_key=options_data,
            root_dir=OPTION_DIR,
        )
    combined_option_dfs.append(pl.read_csv(local_path))


combined_stock_df = pl.concat(combined_stock_dfs)
combined_stock_df = combined_stock_df.with_columns(
    pl.col("window_start")
    .cast(pl.Datetime("ns"))
    .dt.replace_time_zone("UTC")
    .dt.convert_time_zone("America/New_York")
).sort("window_start")

combined_option_df = pl.concat(combined_option_dfs)
combined_option_df = combined_option_df.with_columns(
    pl.col("window_start")
    .cast(pl.Datetime("ns"))
    .dt.replace_time_zone("UTC")
    .dt.convert_time_zone("America/New_York")
)
combined_option_df = clean_and_parse_option_names(combined_option_df)

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


def update_expiry_dates(stock_name: str):
    unique_expiry_dates = sorted(
        stock_and_option.filter(pl.col("symbol") == stock_name)
        .select("expiry_date")
        .unique()
        .to_series()
        .to_list()
    )
    if unique_expiry_dates:
        return unique_expiry_dates
    else:
        return None


# Widgets
stock_options = stock_and_option.select("symbol").unique().to_series().to_list()
stock_name_dropdown = pn.widgets.Select(
    name="Stock Ticker", options=stock_options, value="NVDA"
)
option_type_dropdown = pn.widgets.Select(
    name="Option Type", options={"Put": "P", "Call": "C"}, value="P"
)
expiry_date_dropdown = pn.widgets.Select(
    name="Option Expiry Date", options=update_expiry_dates(stock_name_dropdown.value)
)


def get_histogram_figure(stock_name: str):
    stock = combined_stock_df.filter(pl.col("ticker") == stock_name)
    stock = stock.with_columns(
        (pl.col("close") / pl.col("open")).log().alias("log_return")
    )
    fig = px.histogram(stock["log_return"].to_numpy(), range_x=[-0.01, 0.01])
    fig.update_layout(
        title="Histogram of Daily Log Returns",
        xaxis_title="Log Return",
        yaxis_title="Count",
        bargap=0.1,
        plot_bgcolor=COLOURS["background"],
        paper_bgcolor=COLOURS["background"],
        font_color=COLOURS["text"],
        autosize=True,
    )
    return fig


def get_option_topography_figure(stock_name: str, expiry_date, option_type: str):
    stock_and_option_for_specific_stock = stock_and_option.filter(
        (pl.col("symbol") == stock_name)
        & (pl.col("expiry_date") == expiry_date)
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

    # Interpolate NaNs
    pivot_interp = pivot.copy()
    pivot_interp = pivot_interp.interpolate(axis=1, method="linear")
    pivot_interp = pivot_interp.interpolate(axis=0, method="linear")
    pivot_interp = pivot_interp.bfill(axis=1).ffill(axis=1)

    fig = go.Figure(
        data=[
            go.Surface(
                z=pivot_interp.values,
                x=pivot_interp.columns,
                y=pivot_interp.index.tolist(),
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
        title="% Change in Option Premium / Stock Price Ratio",
        scene=dict(
            xaxis_title="Time",
            yaxis_title="Strike Price",
            zaxis_title="% Change in Ratio",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        autosize=True,
    )
    return fig


def get_range_graph_figure(stock_name: str):
    stock = combined_stock_df.filter(pl.col("ticker") == stock_name)
    stock = stock.with_columns(
        (pl.col("close") / pl.col("open")).log().alias("log_return")
    )
    mean = stock["log_return"].mean()
    vol = stock["log_return"].std()
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
        autosize=True,
    )
    return fig


winners_table = pn.widgets.Tabulator(
    winners.head(20).to_pandas(),
    pagination="remote",
    page_size=20,
)


losers_table = pn.widgets.Tabulator(
    losers.head(20).to_pandas(),
    pagination="remote",
    page_size=20,
)


histogram_pane = pn.bind(
    get_histogram_figure, stock_name=stock_name_dropdown.param.value
)
option_topography_pane = pn.bind(
    get_option_topography_figure,
    stock_name=stock_name_dropdown.param.value,
    expiry_date=expiry_date_dropdown.param.value,
    option_type=option_type_dropdown.param.value,
)

range_graph_pane = pn.bind(
    get_range_graph_figure, stock_name=stock_name_dropdown.param.value
)

gold_sidebar = pn.pane.PNG("https://goldprice.org/charts/gold_1d_o_USD_z.png")
silver_sidebar = pn.pane.PNG("https://goldprice.org/charts/silver_1d_o_USD_z.png")

dashboard = pn.template.MaterialTemplate(
    title="Stock Exploration Dashboard",
    sidebar=[
        stock_name_dropdown,
        expiry_date_dropdown,
        option_type_dropdown,
        gold_sidebar,
        silver_sidebar,
    ],
    main=pn.Tabs(
        ("Option Topography", option_topography_pane),
        ("Histogram", histogram_pane),
        ("Range Graph", range_graph_pane),
        (
            "Winners & Losers",
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("# Winners"),
                    winners_table,
                    sizing_mode="stretch_width",
                ),
                pn.Column(
                    pn.pane.Markdown("# Losers"),
                    losers_table,
                    sizing_mode="stretch_width",
                ),
            ),
        ),
    ),
    theme="dark",
    header_background=COLOURS["background"],
).servable()
