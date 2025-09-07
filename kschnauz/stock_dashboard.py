import logging
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dash_table, dcc, html


pd.options.plotting.backend = "plotly"
pd.set_option("display.max_columns", None)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


app = Dash()

# Body ------------------------------------
COLOURS = {"background": "#000000", "text": "#7FDBFF"}


ROOT_DIR = Path(os.environ.get("ROOT_DIR"))
STOCK_DIR = ROOT_DIR / "kschnauz/stock_csvs"
OPTION_DIR = ROOT_DIR / "kschnauz/options_csvs"
TICKERS = pd.read_csv(f"{ROOT_DIR}/all_tickers.csv")

# "Loose" because Polygon data is sometimes malformed
OCC_OPTION_NAME_PATTERN_LOOSE = re.compile(r"^O:([A-Z]+)(\d{6,7})([CP])(\d{6,9})$")


def clean_option_names(df: pd.DataFrame, column_name: str = "ticker") -> pd.DataFrame:
    """Clean and parse Options Clearing Corporation (OCC) style option names.

    Args:
        df: Input DataFrame containing `column_name`.
        column: Name of the column containing option symbols.

    Returns:
        pd.DataFrame: DataFrame with additional parsed columns:
            ['symbol', 'expiry_date', 'option_type', 'strike_price', 'expiration']
    """
    result = df.copy()

    logger.info(f"Cleaning option names in column '{column_name}'")

    # Extract fields with lenient pattern to allow malformed but recoverable entries
    result[["symbol", "expiry_date", "option_type", "strike_price"]] = result[
        column_name
    ].str.extract(OCC_OPTION_NAME_PATTERN_LOOSE)

    logger.info(f"Parsed {len(result)} option names")

    # Drop rows that couldn't be parsed
    result.dropna(
        subset=["symbol", "expiry_date", "option_type", "strike_price"], inplace=True
    )
    if dropped_rows := len(df) - len(result):
        logger.info(f"Dropped {dropped_rows} rows with unparseable option names")

    # Normalize expiry_date: take last 6 digits (handles 7-digit errors like 2251219 -> 251219)
    result["expiry_date"] = result["expiry_date"].str[-6:]

    logger.info("Normalized expiry_date to last 6 digits")

    # Normalize strike_price: pad to 8 digits, truncate if longer
    result["strike_price"] = result["strike_price"].str.zfill(8).str[:8]
    result["strike_price"] = result["strike_price"].astype(float) / 1000

    logger.info("Normalized strike_price to 8 digits with 3 decimal places")

    result["expiry_date"] = pd.to_datetime(
        result["expiry_date"], format="%y%m%d", errors="coerce"
    )

    logger.info("Converted expiry_date to datetime")

    result.dropna(subset=["expiry_date"], inplace=True)
    if dropped_rows_due_to_date := len(df) - len(result):
        logger.info(
            f"Dropped {dropped_rows_due_to_date - dropped_rows} rows with unparseable option dates"
        )

    return result


# Concatenate all stock DataFrames in the list into a single DataFrame
combined_stock_df = pd.concat(
    [pd.read_csv(file) for file in STOCK_DIR.glob("*.csv.gz")],
    ignore_index=True,
)
combined_stock_df.sort_values(by=["ticker", "window_start"], ascending=[True, False])
combined_stock_df["window_start"] = pd.to_datetime(combined_stock_df["window_start"])

# Combine all option DataFrames in the list into a single DataFrame
combined_option_df = pd.concat(
    [pd.read_csv(file) for file in OPTION_DIR.glob("*.csv.gz")],
    ignore_index=True,
)
combined_option_df.sort_values(by=["ticker", "window_start"], ascending=[True, False])
combined_option_df["window_start"] = pd.to_datetime(combined_option_df["window_start"])

# Polygon has shitty looking data sometimes in terms of ticker structure, so we need to clean it up
combined_option_df = clean_option_names(combined_option_df)

# Go through the combined list of stocks to find the ones with the biggest weekly gains and losses.
stock_watch = [
    "NVDA",
    # "COIN",
    # "CRCL",
    "LLY",
    "TSLA",
    "TQQQ",
    # Brian added a few!
    "MDB",
    "ALAB",
    "PLTR",
    # "AVAV",
]


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


option = combined_option_df[combined_option_df["ticker"].isin(stock_watch)]
stock = combined_stock_df[combined_stock_df["ticker"].isin(stock_watch)]
print(stock)
combined_stk_opt = pd.merge(
    left=stock,
    right=option,
    how="right",
    on=["ticker", "window_start"],
)
winners_and_losers = calculate_winners_and_losers(combined_stock_df, stock_watch)
winners = winners_and_losers[winners_and_losers["type"] == "Winner"]
losers = winners_and_losers[winners_and_losers["type"] == "Loser"]
print(combined_stk_opt)


"""
temp_df = pd.DataFrame(columns = column_names)

for ticker in stock_watch['ticker']:
    temp_df = stock[stock['ticker'] == ticker]
    temp_df['close'] = temp_df['close'].astype(float)
    print(len(temp_df))
    gain_per = (temp_df.close.iloc[len(temp_df) - 1] - temp_df.close.iloc[0]) / temp_df.close.iloc[0]
    min_gain = top_gains['percentage'].min()
    min_loss = top_loss['percentage'].max()
    print(gain_per)
    print(min_gain)
    if gain_per > min_gain:
        top_gains.percentage.iloc[top_gains.idxmin()] = gain_per
        top_gains.ticker.iloc[top_gains.idxmin()] = temp_df.ticker.iloc[0]
    if gain_per < min_loss:
        top_loss.percentage.iloc[top_gains.idxmax()] = gain_per
        top_loss.ticker.iloc[top_gains.idxmax()] = temp_df.ticker.iloc[0]
"""


# App layout will have the drop down list and all calls for graphs
# Generate the drop down list of all unique ticker names and update all graphs when a new ticker is chosen


app.layout = html.Div(
    style={"backgroundColor": COLOURS["background"]},
    children=[
        html.H1(
            children="Stock Selection",
            style={
                "textAlign": "center",
                "color": COLOURS["text"],
            },
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="stk_dropdown",
                    options=[
                        {"label": i, "value": i} for i in stock["ticker"].unique()
                    ],
                    placeholder="Select Stock Ticker...",
                    value="NVDA",
                ),
            ],
            style={"width": "300px", "background-color": "lightblue"},
            # width: 10%,
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="date_dropdown",
                    options=[
                        {"label": i, "value": i} for i in option["strike_date"].unique()
                    ],
                    placeholder="Select Option Date...",
                    # value="1",
                ),
            ],
            style={"width": "300px", "background-color": "lightblue"},
            # width: 10%,
        ),
        html.Br(),
        html.Div(id="my-output"),
        dcc.Graph(id="option_topography", style={"width": "800px", "height": "800px"}),
        dcc.Graph(id="histogram_graph", style={"width": "500px", "height": "500px"}),
        dcc.Graph(id="range_graph", style={"width": "1500px", "height": "500px"}),
        dash_table.DataTable(
            winners.to_dict("records"),
            [{"name": i, "id": i} for i in winners.columns],
        ),
        dash_table.DataTable(
            losers.to_dict("records"),
            [{"name": i, "id": i} for i in losers.columns],
        ),
    ],
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
    stock_opt = combined_stk_opt[
        (combined_stk_opt["ticker"] == f"{input_value1}")
        & (combined_stk_opt["strike_date"] == f"{input_value2}")
    ]
    print(stock_opt)
    print(stock_opt.isnull().values.any())
    y_data = stock_opt["strike_price"].unique().tolist()
    x_data = stock_opt["window_start"].unique().tolist()
    print(len(x_data))
    print(len(y_data))
    # opt_change = (stock_opt.loc[:, "open_y"] - stock_opt.loc[:, "close_y"]) / stock_opt.loc[:, "open_y"]
    # stk_change = (stock_opt.loc[:, "open_x"] - stock_opt.loc[:, "close_x"]) / stock_opt.loc[:, "open_x"]
    z = pd.DataFrame(index=range(len(x_data)), columns=y_data)
    z[:] = 1
    # print(z)
    for strike_price in y_data:
        # print(stock_opt.query('strike_price == @value')['open_y'])
        # z[value] = stock_opt.query('strike_price == @value')['open_y']
        # print(z)
        # print(stock_opt[stock_opt["strike_price"] == value]['open_y'])
        z[strike_price] = stock_opt[stock_opt["strike_price"] == strike_price][
            "open_y"
        ].reset_index(drop=True)
    print(z)
    print(z.isnull().values.any())
    # sh_0, sh_1 = z.shape
    # x, y = x_data.values, y_data.vaules
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

"""
# -*- coding: utf-8 -*-
"""
"""
Created on Wed Aug 13 18:03:09 2025
@author: kbngr
"""
"""
import boto3
from botocore.config import Config
import csv
import glob
import pandas as pd
import numpy as np
import plotly.io as pio
pd.options.plotting.backend = "plotly"
pio.renderers.default='browser'
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


"""
"""
# Initialize a session using your credentials
session = boto3.Session(
  aws_access_key_id='b6e2281d-bd7e-4cac-ba4c-7c0b61448f87',
  aws_secret_access_key='YFKRr0mLFqFpH64QkBaAqliHkziKCALD',
)

# Create a client with your session and specify the endpoint
s3 = session.client(
  's3',
  endpoint_url='https://files.polygon.io',
  config=Config(signature_version='s3v4'),
)

# List Example
# Initialize a paginator for listing objects
paginator = s3.get_paginator('list_objects_v2')

# Choose the appropriate prefix depending on the data you need:
# - 'global_crypto' for global cryptocurrency data
# - 'global_forex' for global forex data
# - 'us_indices' for US indices data
# - 'us_options_opra' for US options (OPRA) data
# - 'us_stocks_sip' for US stocks (SIP) data
prefix = 'us_stocks_sip'  # Example: Change this prefix to match your data need

# List objects using the selected prefix
for page in paginator.paginate(Bucket='flatfiles', Prefix=prefix):
  for obj in page['Contents']:
    print(obj['Key'])

# Copy example
# Specify the bucket name
bucket_name = 'flatfiles'

# Specify the S3 object key name
object_key = 'us_stocks_sip/minute_aggs_v1/2025/08/2025-08-13.csv.gz'

# Specify the local file name and path to save the downloaded file
# This splits the object_key string by '/' and takes the last segment as the file name
local_file_name = object_key.split('/')[-1]

# This constructs the full local file path
local_file_path = './' + local_file_name

# Download the file
s3.download_file(bucket_name, object_key, local_file_path)
"""
"""
# Main program starts here
def Main():

#Functions -------------------------- 



    def convert_date(val):
        new_val = pd.Timestamp(val)
        return (new_val)    




#Body ------------------------------------

    pd.set_option('display.max_columns', None)
    csv_directory = './CSV_FILES/'
    all_csv_files = glob.glob(csv_directory + '*.csv.gz')

    # Create an empty list to store individual DataFrames
    list_of_dfs = []
    
    # Loop through each CSV file and read it into a DataFrame
    for file in all_csv_files:
        df = pd.read_csv(file)
        list_of_dfs.append(df)
    
    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(list_of_dfs, ignore_index=True)
    
    # Display the combined DataFrame (optional)
    print(combined_df.head())


    #Name of file to read, note this will be changed to do real time pulls
    #local_file_name = '2025-08-13.csv.gz'

    #Stock we want to pull
    Stock_Name = "NVDA"
    
    #Format the csv and pull it into a dataframe, prihnts are for QA/QC
    #df = pd.read_csv(local_file_name, header = 0, dtype={
    #    'ticker': str,
    #    'volume': int,
    #    'open': float,
    #    'close' : float,
    #    'high' : float,
    #    'low' : float,
    #    'window_start' : int,
    #    'transactions' : int
    #})
    #print(pd.options.display.max_rows)
    #print(df)
    #print(df.columns)
    #print(df.ticker)
    
    
    #Convert ticker to string as it appears to come in as an object
    combined_df['ticker'] = combined_df['ticker'].astype(str)
    
    #specifi what stock to query, and print for QA then pass to a new data frame for the individual stock
    #query_string_format = f"ticker.str.startswith('{Stock_Name}')"
    #print(combined_df.query(query_string_format))
    
    #stock = combined_df.query(query_string_format)
    
    stock = combined_df[combined_df['ticker'] == f'{Stock_Name}']
    
    #stock.loc[:,('window_start')] = pd.to_datetime(stock.loc[:,('window_start')], unit='ns').astype('datetime64[ns]')
    stock.loc[:, ('window_start')] = stock['window_start'].apply(convert_date)

    #print(stock.query(query_string_format))

    #Plot the open price against time
    fig = stock.plot(x='window_start', y='open')
    fig.update_layout(yaxis_range=[min(stock['open']),max(stock['open'])], xaxis_range=[min(stock['window_start']),max(stock['window_start'])])
    fig.update_layout(title_text= "{} stock price chart".format(Stock_Name),
                      title_font_size=30,
                      xaxis_title_text = "Date",
                      yaxis_title_text = "Price")
    fig.show()

    fig = go.Figure(data=[go.Candlestick(x=stock['window_start'],
                    open=stock['open'],
                    high=stock['high'],
                    low=stock['low'],
                    close=stock['close'])])
    
    fig.show()

    stock['Log_return'] = np.log(stock.loc[:,('close')]/stock.loc[:, ('open')])

    print(stock.head())
    
    fig = px.histogram(stock['Log_return'])
    fig.show()
    
    sns.histplot(stock.loc[:,'Log_return'])
    #plt.show()
    
    mean = np.mean(stock.loc[:,'Log_return'])
    vol = np.std(stock.loc[:,'Log_return'])
    
    mean_annual_log=252*mean
    vol_annual_log=252**0.5*vol
    
    mean_annual_effective=np.exp(mean_annual_log)-1
    vol_annual_effective=np.exp(vol_annual_log)-1
    print(f'The anuual effective return of {Stock_Name} is: {mean_annual_effective*100}%, The annual effective volatility of {Stock_Name} is: {vol_annual_effective*100}%')
    
    SAP = np.average(stock['close'])
    print(f'The close average is: {SAP}%')
    stock['price_dif'] = np.square(stock.loc[:,('close')] - SAP)
    print(stock.head())
    Dif_Sum = np.sum(stock.loc[:,('price_dif')])
    Varience = Dif_Sum / (len(stock['price_dif'] - 1))
    SV = np.sqrt(Varience)
    print(f'The Stock Volitility is: {SV}%')
    #UF = np.exp(SV * np.sqrt(0.00396))
    #print(f'The Up Factor per day is: ${UF}')
    
    
    #from matplotlib.ticker import ScalarFormatter

    mean_daily_effective=np.exp(mean)-1
    vol_daily_effective=np.exp(vol)-1
    first_price=stock.close.iloc[0]
    
    stock['high']=first_price*np.array([((1+mean_daily_effective)**i) * ((1+vol_daily_effective)**np.sqrt(i)) for i in range(len(stock))])
    
    stock['low']=first_price*np.array([((1+mean_daily_effective)**i) / ((1+vol_daily_effective)**np.sqrt(i)) for i in range(len(stock))])
    
    fig = stock.plot(x='window_start', y=['close','high','low'], log_y = True)
    fig.update_yaxes(type="log")
    #fig.yscale('log')
    # Customize the format of the y-axis tick labels
    #plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    #plt.title(f'{Stock_Name} price with confidence intervals')
    fig.show()
    
if __name__ == '__main__':
    Main()
"""
