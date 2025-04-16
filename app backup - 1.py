import dash
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, callback_context
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
import calendar
import datetime
import requests
from io import StringIO

def fetch_ferry_data():
    # To hit our API, you'll be making requests to:
    base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
     
    # Datasets are called "packages". Each package can contain many "resources"
    url = base_url + "/api/3/action/package_show"
    params = { "id": "toronto-island-ferry-ticket-counts"}
    package = requests.get(url, params = params).json()
     
    # To get resource data:
    for idx, resource in enumerate(package["result"]["resources"]):
        if resource["datastore_active"]:
            url = base_url + "/datastore/dump/" + resource["id"]
            resource_dump_data = requests.get(url).text
            return resource_dump_data
    return None

# Load & process data
def load_data():
    resource_dump_data = fetch_ferry_data()
    if resource_dump_data is None:
        raise Exception("Failed to fetch ferry data")
        
    df = pd.read_csv(StringIO(resource_dump_data))
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Year'] = df['Timestamp'].dt.year
    df['Month_Num'] = df['Timestamp'].dt.month
    df['Month'] = df['Month_Num'].apply(lambda x: calendar.month_name[x])
    df['Day'] = df['Timestamp'].dt.date
    df['Hour'] = df['Timestamp'].dt.hour
    df['Minute'] = df['Timestamp'].dt.strftime('%H:%M')
    df['rounded_time'] = df['Timestamp'].dt.floor('h')

    weather_df = pd.read_csv('weather_data.csv')
    weather_df['time'] = pd.to_datetime(weather_df['time'])

    merged = pd.merge(df, weather_df, left_on='rounded_time', right_on='time', how='left')
    return df, weather_df, merged

# App setup
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Initialize global variables
df = None
weather_df = None
merged = None
drill_state = []
drill_levels = ['Year', 'Month', 'Day', 'Hour', 'Minute']
drill_titles = [
    'Total Redemptions by Year',
    'Monthly Redemptions in {}',
    'Daily Redemptions in {} {}',
    'Hourly Redemptions on {}',
    'Minute-Level Redemptions on {}'
]

# Layout
app.layout = dbc.Container([
    dbc.NavbarSimple(brand="Toronto Ferry Dashboard", color="primary", dark=True),
    dcc.Store(id='data-store', data={'last_updated': None}),
    dbc.Row([
        dbc.Col([
            html.H4("Ferry Ticket Redemptions"),
            html.Div(id="last-updated", style={"fontSize": "14px", "color": "black", "marginBottom": "10px"}),
            html.Button("Back", id="back-button", n_clicks=0),
            dcc.Graph(id="bar-graph")
        ], width=8)
    ])
], fluid=True)

# Callback to load data on page load
@app.callback(
    Output('data-store', 'data'),
    Input('data-store', 'data')
)
def load_data_on_refresh(data):
    global df, weather_df, merged
    try:
        df, weather_df, merged = load_data()
        return {'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    except Exception as e:
        print(f"Error loading data: {e}")
        return {'last_updated': None}

# Callback to update last updated time
@app.callback(
    Output('last-updated', 'children'),
    Input('data-store', 'data')
)
def update_last_updated(data):
    if data and data.get('last_updated'):
        return f"Last updated: {data['last_updated']}"
    return "Last updated: Never"

# Graph Update Callback (Drilldown)
@app.callback(
    Output("bar-graph", "figure"),
    Input("bar-graph", "clickData"),
    Input("back-button", "n_clicks"),
    Input("data-store", "data"),
    State("bar-graph", "figure")
)
def update_bar_graph(clickData, n_clicks, data, current_fig):
    global drill_state, df

    if df is None:
        return px.bar(title="Loading data...")

    ctx = callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if trigger == "back-button" and drill_state:
        drill_state.pop()
    elif trigger == "bar-graph" and clickData is not None and len(drill_state) < 4:
        clicked_value = str(clickData['points'][0]['x'])
        drill_state.append(clicked_value)

    level = len(drill_state)

    if level == 0:
        df_grouped = df.groupby("Year")['Redemption Count'].sum().reset_index()
        fig = px.bar(df_grouped, x="Year", y="Redemption Count", title=drill_titles[0])
    elif level == 1:
        df_filtered = df[df['Year'] == int(drill_state[0])]
        df_grouped = df_filtered.groupby(['Month', 'Month_Num'])['Redemption Count'].sum().reset_index()
        df_grouped = df_grouped.sort_values(by='Month_Num')
        fig = px.bar(df_grouped, x="Month", y="Redemption Count", title=drill_titles[1].format(drill_state[0]))
    elif level == 2:
        df_filtered = df[(df['Year'] == int(drill_state[0])) & (df['Month'] == drill_state[1])]
        df_grouped = df_filtered.groupby("Day")['Redemption Count'].sum().reset_index()
        fig = px.bar(df_grouped.sort_values(by="Day"), x="Day", y="Redemption Count", title=drill_titles[2].format(drill_state[1], drill_state[0]))
    elif level == 3:
        df_filtered = df[df['Day'] == datetime.datetime.strptime(drill_state[2], "%Y-%m-%d").date()]
        df_grouped = df_filtered.groupby("Hour")['Redemption Count'].sum().reset_index()
        fig = px.bar(df_grouped, x="Hour", y="Redemption Count", title=drill_titles[3].format(drill_state[2]))
    elif level == 4:
        df_filtered = df[
            (df['Day'] == datetime.datetime.strptime(drill_state[2], "%Y-%m-%d").date()) &
            (df['Hour'] == int(drill_state[3]))
        ]
        df_grouped = df_filtered.groupby("Timestamp")['Redemption Count'].sum().reset_index()
        df_grouped["label"] = df_grouped["Timestamp"].dt.strftime('%H:%M')
        fig = px.bar(df_grouped, x="label", y="Redemption Count", title=drill_titles[4].format(drill_state[2]))
    else:
        return dash.no_update
    
    fig.update_xaxes(type='category')
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)
