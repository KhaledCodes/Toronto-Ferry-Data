from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, callback_context
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
import calendar
import requests
import datetime

# Incorporate data
df = pd.read_csv('data.csv')

# Create Date Columns
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month.apply(lambda x: calendar.month_name[int(x)])
df['Day'] = df['Timestamp'].dt.date
df['Hour'] = df['Timestamp'].dt.hour
df['Minute'] = df['Timestamp'].dt.strftime('%H:%M')

# Incorporate weather data
weather_df = pd.read_csv('weather_data.csv')

# Rounded time
df['rounded_time'] = df['Timestamp'].dt.floor('h')

# Change time data type
weather_df['time'] = pd.to_datetime(weather_df['time'])

# Join 2 datasets
merged = pd.merge(df, weather_df, left_on='rounded_time', right_on='time', how='left')


# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = dbc.Container([
    dbc.NavbarSimple(brand="Toronto Ferry Dashboard", color="primary", dark=True),
    dbc.Row([
        dbc.Col([
            html.H4("Redemption Counts"),
            html.Button("Back", id="back-button", n_clicks=0),
            dcc.Graph(id="bar-graph")
        ], width=8)
        # dbc.Col([
        #     html.H4("Top 10 Days"),
        #     dash_table.DataTable(data=day_grouped_sorted.to_dict('records'), page_size=10),
        # ], width=4),
        # dbc.Col([
        #     html.H4("Top Month"),
        #     dbc.Card([
        #         dbc.CardBody(
        #             [
        #                 html.H3(top_month, className ="card-title"),
        #                 html.P(int(top_month_rd), className="card-text")
        #             ]
        #         )
        #     ],
        #         style={"width": "10rem"},
        #     )
        # ])
    ])
], fluid=True)

# App state to track drilldown level
drill_levels = ['Year', 'Month', 'Day', 'Hour', 'Minute']
drill_titles = [
    'Total Redemptions by Year',
    'Monthly Redemptions in {}',
    'Daily Redemptions in {} {}',
    'Hourly Redemptions on {}',
    'Minute-Level Redemptions on {}'
]

drill_state = []

@app.callback(
    Output("bar-graph", "figure"),
    Input("bar-graph", "clickData"),
    Input("back-button", "n_clicks"),
    State("bar-graph", "figure")
)
def update_graph(clickData, n_clicks, current_fig):
    global drill_state
    ctx = callback_context

    if not ctx.triggered:
        trigger = None
    else:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == "back-button" and drill_state:
        drill_state.pop()

    elif trigger == "bar-graph" and clickData is not None and len(drill_state) < 4:
        clicked_value = str(clickData['points'][0]['x'])
        drill_state.append(clicked_value)

    # Determine drilldown level
    level = len(drill_state)

    if level == 0:
        df_grouped = df.groupby("Year")['Redemption Count'].sum().reset_index()
        fig = px.bar(df_grouped, x="Year", y="Redemption Count", title=drill_titles[0])
        fig.update_xaxes(type='category')
        return fig

    elif level == 1:
        df_filtered = df[df['Year'] == int(drill_state[0])]
        df_grouped = df_filtered.groupby("Month")['Redemption Count'].sum().reset_index()
        fig = px.bar(df_grouped.sort_values(by="Month"), x="Month", y="Redemption Count", title=drill_titles[1].format(drill_state[0]))
        fig.update_xaxes(type='category')
        return fig

    elif level == 2:
        df_filtered = df[(df['Year'] == int(drill_state[0])) & (df['Month'] == drill_state[1])]
        df_grouped = df_filtered.groupby("Day")['Redemption Count'].sum().reset_index()
        fig = px.bar(df_grouped.sort_values(by="Day"), x="Day", y="Redemption Count", title=drill_titles[2].format(drill_state[1], drill_state[0]))
        fig.update_xaxes(type='category')
        return fig

    elif level == 3:
        df_filtered = df[df['Day'] == datetime.datetime.strptime(drill_state[2], "%Y-%m-%d").date()]
        df_grouped = df_filtered.groupby("Hour")['Redemption Count'].sum().reset_index()
        fig = px.bar(df_grouped, x="Hour", y="Redemption Count", title=drill_titles[3].format(drill_state[2]))
        fig.update_xaxes(type='category')
        return fig

    elif level == 4:
        df_filtered = df[
            (df['Day'] == datetime.datetime.strptime(drill_state[2], "%Y-%m-%d").date()) &
            (df['Hour'] == int(drill_state[3]))
    ]
        df_grouped = df_filtered.groupby("Timestamp")['Redemption Count'].sum().reset_index()
        df_grouped["label"] = df_grouped["Timestamp"].dt.strftime('%H:%M')
        fig = px.bar(df_grouped, x="label", y="Redemption Count", title=drill_titles[4].format(drill_state[2]))
        fig.update_xaxes(type='category')
        return fig

    return dash.no_update

# Run the app
server = app.server

if __name__ == '__main__':
    app.run(debug=True,port=8051)
