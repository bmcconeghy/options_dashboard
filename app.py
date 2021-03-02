import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.express as px
import pandas as pd
from yahoo_fin import options, stock_info

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

all_tickers = pd.read_csv('all_tickers.csv')

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


def get_ticker_options(ticker_table):
    tickers = []
    for pos, symbol in ticker_table.iterrows():
        tickers.append({'label': f'{ticker_table["Symbol"][pos]} - {ticker_table["Name"][pos]}', 
                        'value': ticker_table["Symbol"][pos]})
    return tickers


app.layout = html.Div(style={'backgroundColor': colors['background'], 
                            'color': colors['text']},
                      children=[
    html.H1('Options Dashboard', style={'textAlign': 'center'}),
                      
    html.Label('Stock Ticker'),
    dcc.Dropdown(
        id='ticker',
        options=get_ticker_options(all_tickers),
        value='NOK'),
                          
    html.Label('Option Expiry Date'),
    dcc.Dropdown(id='dates', multi=True),

    html.Div([
        dcc.Graph(id='options-price-graph', style={'display': 'inline-block', 'flex': '50%'}),
        dcc.Graph(id='stock-price-graph', style={'display': 'inline-block', 'flex': '50%'})
    ])
                          
])
    

@app.callback(
    Output('options-price-graph', 'figure'),
    Input('ticker', 'value'),
    Input('dates', 'value')
)
def update_option_plot(ticker: str, dates: list):
    ticker = ticker.upper()
    price = stock_info.get_live_price(ticker)
    if dates:
        dfs = []
        for d in dates:
            df = options.get_calls(ticker, d)
            df['Expiry Date'] = d
            dfs.append(df)
        df = pd.concat(dfs)
    else:
        df = options.get_calls(ticker)
        df['Expiry Date'] = 'This Friday'

    fig = px.scatter(df, 
                     x="Strike",
                     y="Last Price",
                     title=ticker,
                     size='Open Interest',
                     color='Expiry Date',
                     hover_data=['Open Interest', 'Implied Volatility']).update_traces(mode='lines+markers', marker_line_width=2)
    
    fig.add_vline(price, line_dash='dash', line_color='green', annotation_text=f'Current Stock Price: ${round(price, 2)}')
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    return fig


@app.callback(
    Output(component_id='dates', component_property='options'),
    Input(component_id='ticker', component_property='value')
)
def populate_dates(ticker):
    all_dates = options.get_expiration_dates(ticker)
    dates = []
    for date in all_dates:
        dates.append({'label': date, 'value': date})
    return dates


@app.callback(
    Output('stock-price-graph', 'figure'),
    Input('ticker', 'value')
)
def update_stock_plot(ticker: str):
    now = pd.Timestamp.today()
    df = stock_info.get_data(ticker, start_date=now - pd.DateOffset(days=1), end_date=now + pd.DateOffset(1), interval='1m', index_as_date=False).dropna()
    df['date'] = df['date'].apply(lambda x: x - pd.DateOffset(hours=8))
    fig = px.scatter(df, x='date', y='close', size='volume', trendline='lowess', title="Today's Stock Price")
    fig.update_layout(
        xaxis_title='Date', yaxis_title='Price',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)