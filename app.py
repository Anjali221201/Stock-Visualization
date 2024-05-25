import dash  
from dash import dcc, html
from dash.dependencies import Input, Output, State
from datetime import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

app = dash.Dash(__name__, external_stylesheets=['/assets/styles.css'])
server = app.server

# Layout Components
item1 = html.Div(
    [
        html.Div(
            [html.P("Welcome!!", className="welcome")],
        ),
        html.Div(
            [html.P("to the stock Dash App!", className="bgtitle")],
        ),
        html.Div(
            [
                dcc.Input(id='stock-code-input', type='text', placeholder='Enter stock code', value='AAPL', className="stock-code-input"),
                html.Button('Submit', id='submit-button', n_clicks=0, className="submit-button"),
            ]
        ),
        html.Div(
            [
                dcc.DatePickerRange(
                    id='date-picker-range',
                    start_date=None,
                    end_date=dt.today(),
                    max_date_allowed=dt.today(),
                    display_format='DD-MM-YYYY',
                    className="date-picker-range"
                )
            ]
        ),
        html.Div(
            [
                html.Button('Get Stock Price', id='stock-price-button', n_clicks=0, className="stock-price-button"),
                html.Button('Get Indicators', id='indicators-button', n_clicks=0, className="indicators-button"),
                dcc.Input(id='forecast-days-input', type='number', placeholder='Enter no. of days', min=1, max=30, className="forecast-days-input"),
                html.Button('Run', id='forecasting-button', n_clicks=0, className="forecasting-button"),
                html.Div(id='forecast-results', className="forecast-results"),
            ],
        ),
    ],
    className="nav"
)

item2 = html.Div(
    [
        html.A(
            "Don't know the stock code? Click here for help.",
            href="/assets/stock_codes.pdf",
            target="_blank",
            className="pdf-link"
        ),
       
        html.Div(
            [
                html.Img(src='assets/Group 2LOGO.png', alt='Company Logo', className='logo'),
                html.H1('Quantum stocks', id='company-name', className='company-name')
            ],
            className="header"
        ),
        ###############
        html.Div(
            [html.Div("Description Placeholder", id="description", className="description-ticker")]
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Stock Price", className="graph-title"),
                        dcc.Graph(id='stock-price-plot',className="graph")
                    ],
                    className="graph-container"
                )
            ],
            id="graphs-content"
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Indicators", className="graph-title"),
                        dcc.Graph(id='indicator-plot',className="graph")
                    ],
                    className="graph-container"
                )
            ],
            id="main-content"
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Forecast", className="graph-title"),
                        dcc.Graph(id='forecast-plot',className="graph")
                    ],
                    className="graph-container"
                )
            ],
            id="forecast-content"
        )
    ],
    className="content"
)

app.layout = html.Div([item1, item2], className="container")

# Callback for company name
@app.callback(
    Output('company-name', 'children'),
    Input('submit-button', 'n_clicks'),
    State('stock-code-input', 'value')
)
def update_company_name(n_clicks, stock_code):
    if stock_code:
        try:
            ticker = yf.Ticker(stock_code)
            info = ticker.info
            company_name = info.get("longName", "Quantum Stocks")
            return company_name
        except Exception as e:
            return "Error retrieving company name"
    else:
        return "Quantum stocks"


# Callback for company description
@app.callback(
    Output('description', 'children'),
    Input('submit-button', 'n_clicks'),
    State('stock-code-input', 'value')
)
def get_data(n_clicks, input1):
    if input1:
        try:
            ticker = yf.Ticker(input1)
            inf = ticker.info
            if 'longBusinessSummary' in inf:
                return inf["longBusinessSummary"]
            else:
                return "No business summary available for the specified stock."
        except Exception as e:
            return f"Error retrieving data: {e}"
    else:
        return "Please enter a valid stock code."

# Function to generate stock price plot
def get_stock_price_fig(df):
    fig = px.line(df, x='Date', y=['Open', 'Close'], title="Closing and Opening Price vs Date")
    return fig

# Callback for generating stock price graph
@app.callback(
    Output('stock-price-plot', 'figure'),
    Input('stock-price-button', 'n_clicks'),
    State('stock-code-input', 'value'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date')
)
def update_stock_price_plot(n_clicks, stock_code, start_date, end_date):
    if n_clicks > 0 and stock_code and start_date and end_date:
        try:
            df = yf.download(stock_code, start=start_date, end=end_date)
            if df.empty:
                return go.Figure()  # Return an empty figure if no data
            df.reset_index(inplace=True)
            fig = get_stock_price_fig(df)
            return fig
        except Exception as e:
            return go.Figure()
    else:
        return go.Figure()

# Function to generate indicator plot
def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df, x=df.index, y="EWA_20", title="Exponential Moving Average vs Date")
    fig.update_traces(mode='markers+lines')
    return fig

# Callback for indicator plot
@app.callback(
    Output('indicator-plot', 'figure'),
    Input('indicators-button', 'n_clicks'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date'),
    State('stock-code-input', 'value')
)
def update_indicator_plot(n_clicks, start_date, end_date, stock_code):
    if n_clicks > 0 and start_date and end_date and stock_code:
        try:
            df = yf.download(stock_code, start=start_date, end=end_date)
            if df.empty:
                return go.Figure()  # Return an empty figure if no data
            df.reset_index(inplace=True)
            fig = get_more(df)
            return fig
        except Exception as e:
            return go.Figure()
    else:
        return go.Figure()

# Callback for forecast plot
@app.callback(
    Output('forecast-plot', 'figure'),
    Input('forecasting-button', 'n_clicks'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date'),
    State('forecast-days-input', 'value'),
    State('stock-code-input', 'value')
)
def update_forecast_plot(n_clicks, start_date, end_date, input_days, stock_code):
    if n_clicks > 0 and input_days and stock_code:
        try:
            test_start = pd.to_datetime(start_date)
            test_end = pd.to_datetime(end_date) + pd.DateOffset(days=int(input_days))

            test_data = yf.download(stock_code, test_start, test_end)
            if test_data.empty:
                return go.Figure()  # Return an empty figure if no data
            test_data.reset_index(inplace=True)

            X = np.arange(len(test_data)).reshape(-1, 1)
            y = test_data['Close'].values

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

            svr_model = SVR(kernel='linear')
            svr_model.fit(X_scaled, y_scaled)

            forecast_X = np.arange(len(test_data), len(test_data) + int(input_days)).reshape(-1, 1)
            forecast_scaled = svr_model.predict(scaler_X.transform(forecast_X))

            forecast = scaler_y.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

            forecast_dates = pd.date_range(start=test_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=int(input_days), freq='D')
            forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted': forecast})

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], mode='lines+markers', name='Observed'))
            fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecasted'], mode='lines+markers', name='Forecasted'))

            fig.update_layout(title="Observed vs Predicted Prices", xaxis_title="Date", yaxis_title="Stock Price", legend_title="Legend")
            return fig
        except Exception as e:
            return go.Figure()
    else:
        return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)