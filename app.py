import dash
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, callback_context
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import calendar
import datetime
import requests
from io import StringIO
from datetime import datetime, timedelta
from functools import lru_cache
import os
import time

# Global variable to store the last refresh time and data
# This is okay to be global as it's read-only for users
last_refresh = {
    'timestamp': None,
    'ferry_data': None,
    'weather_data': None,
    'last_backup_date': None
}

# Constants
HISTORICAL_DATA_FILE = 'historical_ferry_data.csv'
DATA_BACKUP_FILE = 'data_backup.csv'

def save_to_historical_data(df):
    """Save new data to historical dataset, avoiding duplicates"""
    try:
        print("\nDEBUG - Starting save_to_historical_data")
        print(f"DEBUG - Input data shape: {df.shape}")
        print(f"DEBUG - Input date range: {df['Day'].min()} to {df['Day'].max()}")
        
        # Load existing historical data if it exists
        if os.path.exists(HISTORICAL_DATA_FILE):
            print(f"DEBUG - Loading existing historical data from {HISTORICAL_DATA_FILE}")
            historical_df = pd.read_csv(HISTORICAL_DATA_FILE)
            historical_df['Timestamp'] = pd.to_datetime(historical_df['Timestamp'])
            historical_df['Day'] = pd.to_datetime(historical_df['Timestamp']).dt.date
            
            print(f"DEBUG - Existing historical data shape: {historical_df.shape}")
            print(f"DEBUG - Existing historical date range: {historical_df['Day'].min()} to {historical_df['Day'].max()}")
            
            # Merge new data with historical data, keeping only unique records
            print("DEBUG - Merging new data with historical data")
            combined_df = pd.concat([historical_df, df]).drop_duplicates(subset=['Timestamp'])
            combined_df = combined_df.sort_values('Timestamp')
            
            print(f"DEBUG - Combined data shape: {combined_df.shape}")
            print(f"DEBUG - Combined date range: {combined_df['Day'].min()} to {combined_df['Day'].max()}")
        else:
            print("DEBUG - No existing historical data found, using input data")
            combined_df = df
        
        # Save the combined data
        print(f"DEBUG - Saving data to {HISTORICAL_DATA_FILE}")
        combined_df.to_csv(HISTORICAL_DATA_FILE, index=False)
        
        # Create a backup
        print(f"DEBUG - Creating backup in {DATA_BACKUP_FILE}")
        combined_df.to_csv(DATA_BACKUP_FILE, index=False)
        
        print(f"DEBUG - Successfully updated historical data. Total records: {len(combined_df)}")
        return combined_df
    except Exception as e:
        print(f"Error saving historical data: {e}")
        import traceback
        print(traceback.format_exc())
        return df

@lru_cache(maxsize=1)
def get_processed_data():
    """Get processed data from both API and historical sources"""
    data = load_data()
    df = pd.read_json(StringIO(data['ferry_data']), orient='split')
    
    # Ensure proper date handling
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Day'] = pd.to_datetime(df['Timestamp']).dt.date
    df['Year'] = df['Timestamp'].dt.year
    df['Month_Num'] = df['Timestamp'].dt.month
    df['Month'] = df['Month_Num'].apply(lambda x: calendar.month_name[x])
    df['Day_Str'] = df['Day'].astype(str)
    df['Hour'] = df['Timestamp'].dt.hour
    df['Minute'] = df['Timestamp'].dt.strftime('%H:%M')
    df['rounded_time'] = df['Timestamp'].dt.floor('h')

    # Save to historical data once per day
    current_time = datetime.now()
    if last_refresh.get('last_backup_date') != current_time.date():
        df = save_to_historical_data(df)
        last_refresh['last_backup_date'] = current_time.date()
    
    return df

def is_complete_year(df, year):
    """Check if a year has data for the full year (Jan 1 to Dec 31)"""
    year_data = df[df['Year'] == year]
    if year_data.empty:
        return False
    
    # For current year, consider it complete if we have data from Jan 1
    if year == datetime.now().year:
        first_day = min(year_data['Day'])
        return first_day == datetime(year, 1, 1).date()
    
    # For past years, check if we have full year of data
    first_day = min(year_data['Day'])
    last_day = max(year_data['Day'])
    
    # Special handling for 2015 - exclude it since data starts from May
    if year == 2015:
        return False
    
    # For all other past years, require complete data (Jan 1 to Dec 31)
    return (first_day == datetime(year, 1, 1).date() and 
            last_day == datetime(year, 12, 31).date())

def fetch_ferry_data():
    # To hit our API, you'll be making requests to:
    base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
     
    # Datasets are called "packages". Each package can contain many "resources"
    url = base_url + "/api/3/action/package_show"
    params = { "id": "toronto-island-ferry-ticket-counts"}
    
    try:
        # Add a small delay to avoid rate limiting
        time.sleep(1)
        
        print("\nDEBUG - Fetching package info...")
        response = requests.get(url, params = params)
        
        # Check for rate limiting
        if response.status_code == 429:  # Too Many Requests
            print("DEBUG - Rate limited. Waiting 60 seconds...")
            time.sleep(60)
            response = requests.get(url, params = params)
        
        response.raise_for_status()  # Raise an exception for bad status codes
        package = response.json()
        
        # To get resource data:
        for idx, resource in enumerate(package["result"]["resources"]):
            if resource["datastore_active"]:
                url = base_url + "/datastore/dump/" + resource["id"]
                print(f"\nDEBUG - Fetching resource {idx} data from:", url)
                
                # Add a small delay before the second request
                time.sleep(1)
                
                response = requests.get(url)
                
                # Check for rate limiting
                if response.status_code == 429:  # Too Many Requests
                    print("DEBUG - Rate limited. Waiting 60 seconds...")
                    time.sleep(60)
                    response = requests.get(url)
                
                response.raise_for_status()
                return response.text
                
    except requests.exceptions.RequestException as e:
        print(f"DEBUG - Error fetching data: {e}")
        # If we have cached data, return it
        if last_refresh['ferry_data'] is not None:
            print("DEBUG - Using cached data due to API error")
            return last_refresh['ferry_data']
        raise Exception(f"Failed to fetch ferry data: {e}")
    
    return None

def load_data():
    """Load and process data from both API and historical sources"""
    global last_refresh
    
    print("\nDEBUG - Starting load_data()")
    # Check if we need to refresh (if it's been more than an hour or first load)
    current_time = datetime.now()
    if (last_refresh['timestamp'] is None or 
        current_time - last_refresh['timestamp'] >= timedelta(hours=1)):
        
        try:
            print("DEBUG - Fetching new data from API")
            # Try to fetch new data from API
            resource_dump_data = fetch_ferry_data()
            if resource_dump_data is None:
                raise Exception("Failed to fetch ferry data")
            
            print("DEBUG - Converting API data to DataFrame")
            # Convert API data to DataFrame
            df = pd.read_csv(StringIO(resource_dump_data))
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Day'] = pd.to_datetime(df['Timestamp']).dt.date
            
            print(f"DEBUG - API data shape: {df.shape}")
            print(f"DEBUG - API data date range: {df['Day'].min()} to {df['Day'].max()}")
            
            # Load historical data if it exists
            if os.path.exists(HISTORICAL_DATA_FILE):
                print(f"DEBUG - Loading historical data from {HISTORICAL_DATA_FILE}")
                historical_df = pd.read_csv(HISTORICAL_DATA_FILE)
                historical_df['Timestamp'] = pd.to_datetime(historical_df['Timestamp'])
                historical_df['Day'] = pd.to_datetime(historical_df['Timestamp']).dt.date
                
                print(f"DEBUG - Historical data shape: {historical_df.shape}")
                print(f"DEBUG - Historical data date range: {historical_df['Day'].min()} to {historical_df['Day'].max()}")
                
                # Merge API data with historical data
                print("DEBUG - Merging API and historical data")
                df = pd.concat([historical_df, df]).drop_duplicates(subset=['Timestamp'])
            
            # Process the combined data
            df = df.sort_values('Timestamp')
            df['Year'] = df['Timestamp'].dt.year
            df['Month_Num'] = df['Timestamp'].dt.month
            df['Month'] = df['Month_Num'].apply(lambda x: calendar.month_name[x])
            df['Day_Str'] = df['Day'].astype(str)
            df['Hour'] = df['Timestamp'].dt.hour
            df['Minute'] = df['Timestamp'].dt.strftime('%H:%M')
            df['rounded_time'] = df['Timestamp'].dt.floor('h')
            
            print(f"DEBUG - Final data shape: {df.shape}")
            print(f"DEBUG - Final date range: {df['Day'].min()} to {df['Day'].max()}")
            print(f"DEBUG - Years in data: {sorted(df['Year'].unique())}")
            
            # Update the global last_refresh
            last_refresh = {
                'timestamp': current_time,
                'ferry_data': df.to_json(date_format='iso', orient='split'),
                'weather_data': None
            }
            
            # Clear the cache when new data is loaded
            if hasattr(get_processed_data, 'cache_clear'):
                get_processed_data.cache_clear()
            
        except Exception as e:
            print(f"Error refreshing data: {e}")
            import traceback
            print(traceback.format_exc())
            # Keep the old data if refresh fails
            if last_refresh['ferry_data'] is None:
                raise Exception("No data available")
    else:
        print("DEBUG - Using cached data")
    
    return last_refresh

# App setup
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Important: This is what Render needs
server = app.server

# Drill-down titles (this is fine as a constant)
drill_titles = [
    'Total Redemptions by Year',
    'Monthly Redemptions in {}',
    'Daily Redemptions in {} {}',
    'Hourly Redemptions on {}',
    'Minute-Level Redemptions on {}'
]

# Layout
app.layout = dbc.Container([
    # Navigation bar
    dbc.NavbarSimple(brand="Toronto Ferry Dashboard", color="primary", dark=True, className="mb-2"),
    
    # Last updated text below navbar
    html.Div(id="last-updated", className="text-muted small mb-4"),
    
    # Store components
    dcc.Store(id='session-data', data={'drill_state': []}),
    dcc.Store(id='ytd-session-data', data={'drill_state': []}),
    dcc.Interval(id='interval-component', interval=3600000, n_intervals=0),
    
    # Main content
    dbc.Row([
        # Left column - Graphs
        dbc.Col([
            # Total Redemptions Graph
            dbc.Card([
                dbc.CardBody([
                    html.H4("Ferry Ticket Redemptions", className="card-title"),
                    html.Button("Back", id="back-button", className="btn btn-outline-primary mb-3"),
                    dcc.Loading(
                        id="loading-graph",
                        type="circle",
                        children=[
                            dcc.Graph(
                                id="bar-graph",
                                config={'responsive': True},
                                style={'height': '50vh'}  # 50% of viewport height
                            )
                        ]
                    )
                ])
            ], className="mb-4"),
            
            # YTD/MTD Analysis Graph
            dbc.Card([
                dbc.CardBody([
                    html.H4("YTD/MTD Analysis", className="card-title"),
                    html.Button("Back", id="ytd-back-button", className="btn btn-outline-primary mb-3"),
                    dcc.Loading(
                        id="loading-ytd-graph",
                        type="circle",
                        children=[
                            dcc.Graph(
                                id="ytd-bar-graph",
                                config={'responsive': True},
                                style={'height': '50vh'}  # 50% of viewport height
                            )
                        ]
                    )
                ])
            ])
        ], xs=12, md=8),  # Full width on small screens, 8/12 on medium and up
        
        # Right column - Statistics Cards
        dbc.Col([
            html.H4("Visitor Statistics", className="mt-2 mb-4"),
            # YTD Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Last Year (YTD)"),
                        dbc.CardBody([
                            html.H3(id="year-count", className="text-center mb-0")
                        ])
                    ], className="h-100")
                ], xs=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("This Year (YTD)"),
                        dbc.CardBody([
                            html.H3(id="year-comparison", className="text-center mb-0")
                        ])
                    ], className="h-100")
                ], xs=6)
            ], className="mb-4"),
            
            # MTD Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Last Month (MTD)"),
                        dbc.CardBody([
                            html.H3(id="month-count", className="text-center mb-0")
                        ])
                    ], className="h-100")
                ], xs=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("This Month (MTD)"),
                        dbc.CardBody([
                            html.H3(id="month-comparison", className="text-center mb-0")
                        ])
                    ], className="h-100")
                ], xs=6)
            ], className="mb-4"),
            
            # DTD Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("This Day Last Year (DTD)"),
                        dbc.CardBody([
                            html.H3(id="day-count", className="text-center mb-0")
                        ])
                    ], className="h-100")
                ], xs=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Today (DTD)"),
                        dbc.CardBody([
                            html.H3(id="day-comparison", className="text-center mb-0")
                        ])
                    ], className="h-100")
                ], xs=6)
            ])
        ], xs=12, md=4)  # Full width on small screens, 4/12 on medium and up
    ])
], fluid=True, className="px-4 py-3")

# Callback to check for data updates
@app.callback(
    Output('last-updated', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_last_updated(n):
    if last_refresh['timestamp']:
        # Clear the cache if data needs refresh
        current_time = datetime.now()
        if current_time - last_refresh['timestamp'] >= timedelta(hours=1):
            get_processed_data.cache_clear()
        return f"Data last updated: {last_refresh['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
    return "Data last updated: Never"

# Callback to update visitor statistics
@app.callback(
    [Output('year-count', 'children'),
     Output('month-count', 'children'),
     Output('day-count', 'children'),
     Output('year-comparison', 'children'),
     Output('month-comparison', 'children'),
     Output('day-comparison', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_visitor_stats(n):
    try:
        df = get_processed_data()
        
        # Get current date info
        current_date = datetime.now().date()
        current_year = current_date.year
        current_month = current_date.month
        current_day = current_date.day
        
        # Calculate YTD counts
        ytd_current = df[
            (df['Year'] == current_year) & 
            (df['Day'] <= current_date)
        ]['Redemption Count'].sum()
        
        ytd_last_year = df[
            (df['Year'] == current_year - 1) & 
            (df['Day'] <= datetime(current_year - 1, current_month, current_day).date())
        ]['Redemption Count'].sum()
        
        # Calculate MTD counts
        mtd_current = df[
            (df['Year'] == current_year) & 
            (df['Month_Num'] == current_month) &
            (df['Day'] <= current_date)
        ]['Redemption Count'].sum()
        
        # For last year's MTD, handle the case where current month is earlier than last year's data
        last_year_data = df[df['Year'] == current_year - 1]
        if not last_year_data.empty and current_month <= last_year_data['Month_Num'].max():
            mtd_last_year = last_year_data[
                (last_year_data['Month_Num'] == current_month) &
                (last_year_data['Day'] <= datetime(current_year - 1, current_month, current_day).date())
            ]['Redemption Count'].sum()
        else:
            mtd_last_year = 0
        
        # Calculate DTD counts
        dtd_current = df[
            (df['Year'] == current_year) & 
            (df['Month_Num'] == current_month) &
            (df['Day'] == current_date)
        ]['Redemption Count'].sum()
        
        # For last year's DTD, handle the case where the date exists in last year's data
        last_year_same_day = datetime(current_year - 1, current_month, current_day).date()
        dtd_last_year = df[
            (df['Year'] == current_year - 1) & 
            (df['Day'] == last_year_same_day)
        ]['Redemption Count'].sum()
        
        # Calculate percentage changes
        ytd_change = ((ytd_current - ytd_last_year) / ytd_last_year * 100) if ytd_last_year > 0 else 0
        mtd_change = ((mtd_current - mtd_last_year) / mtd_last_year * 100) if mtd_last_year > 0 else 0
        dtd_change = ((dtd_current - dtd_last_year) / dtd_last_year * 100) if dtd_last_year > 0 else 0
        
        # Format the comparison strings with colors
        def format_comparison(current, last_year, change):
            if last_year == 0:
                return html.Span("No data for comparison", style={"textAlign": "center"})
            color = "green" if change >= 0 else "red"
            arrow = "↑" if change >= 0 else "↓"
            return html.Span([
                f"{current:,} ",  # Show current value
                html.Span(f"{arrow} {abs(change):.1f}%", style={"color": color})
            ], style={"textAlign": "center"})
        
        # Format current values
        def format_current(value):
            return f"{value:,}" if value > 0 else "0"
        
        return (
            format_current(ytd_last_year),  # Last Year
            format_current(mtd_last_year),  # Last Month
            format_current(dtd_last_year),  # Yesterday
            format_comparison(ytd_current, ytd_last_year, ytd_change),  # This Year with comparison
            format_comparison(mtd_current, mtd_last_year, mtd_change),  # This Month with comparison
            format_comparison(dtd_current, dtd_last_year, dtd_change)   # Today with comparison
        )
    except Exception as e:
        print(f"Error updating visitor stats: {e}")
        import traceback
        print(traceback.format_exc())
        return "0", "0", "0", "No data for comparison", "No data for comparison", "No data for comparison"

# Graph Update Callback (Drilldown)
@app.callback(
    Output("bar-graph", "figure"),
    Output('session-data', 'data'),
    Input("bar-graph", "clickData"),
    Input("back-button", "n_clicks"),
    Input('interval-component', 'n_intervals'),
    State('session-data', 'data')
)
def update_bar_graph(clickData, n_clicks, n_intervals, session_data):
    try:
        df = get_processed_data()
        drill_state = session_data.get('drill_state', [])

        ctx = callback_context
        trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        if trigger == "back-button" and drill_state:
            drill_state.pop()
        elif trigger == "bar-graph" and clickData is not None and len(drill_state) < 4:
            clicked_value = str(clickData['points'][0]['x'])
            drill_state.append(clicked_value)

        session_data['drill_state'] = drill_state
        level = len(drill_state)

        # Create figure
        fig = go.Figure()

        if level == 0:
            # Filter out 2015 and group by year
            df_filtered = df[df['Year'] != 2015]
            df_grouped = df_filtered.groupby("Year")['Redemption Count'].sum().reset_index()
            
            # Calculate average excluding current year
            current_year = datetime.now().year
            avg_count = df_grouped[df_grouped['Year'] != current_year]['Redemption Count'].mean()
            
            # Add bar chart
            fig.add_trace(go.Bar(
                x=df_grouped['Year'],
                y=df_grouped['Redemption Count'],
                name='Redemption Count',
                text=df_grouped['Redemption Count'].apply(lambda x: f'{x:,.0f}'),  # Add formatted numbers
                textposition='outside'  # Show labels above bars
            ))
            
            # Add average line
            fig.add_trace(go.Scatter(
                x=df_grouped['Year'],
                y=[avg_count] * len(df_grouped),
                mode='lines',
                name=f'Average ({avg_count:,.0f})',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=drill_titles[0],
                margin=dict(t=100)  # Add top margin for labels
            )
            
        elif level == 1:
            selected_year = int(drill_state[0])
            df_filtered = df[df['Year'] == selected_year]
            df_grouped = df_filtered.groupby(['Month', 'Month_Num'])['Redemption Count'].sum().reset_index()
            df_grouped = df_grouped.sort_values(by='Month_Num')
            
            # Calculate average excluding current month if it's the current year
            current_year = datetime.now().year
            current_month = datetime.now().month
            if int(drill_state[0]) == current_year:
                avg_count = df_grouped[df_grouped['Month_Num'] != current_month]['Redemption Count'].mean()
            else:
                avg_count = df_grouped['Redemption Count'].mean()
            
            # Add bar chart
            fig.add_trace(go.Bar(
                x=df_grouped['Month'],
                y=df_grouped['Redemption Count'],
                name='Redemption Count'
            ))
            
            # Add average line
            fig.add_trace(go.Scatter(
                x=df_grouped['Month'],
                y=[avg_count] * len(df_grouped),
                mode='lines',
                name=f'Average ({avg_count:,.0f})',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(title=drill_titles[1].format(drill_state[0]))
            
        elif level == 2:
            df_filtered = df[(df['Year'] == int(drill_state[0])) & (df['Month'] == drill_state[1])]
            df_grouped = df_filtered.groupby("Day")['Redemption Count'].sum().reset_index()
            # Sort by date
            df_grouped = df_grouped.sort_values(by="Day")
            # Convert date to string for display
            df_grouped["Day_Str"] = df_grouped["Day"].astype(str)
            
            # Calculate average excluding current day if it's the current month and year
            current_date = datetime.now().date()
            current_year = datetime.now().year
            current_month = datetime.now().month
            if (int(drill_state[0]) == current_year and 
                calendar.month_name[current_month] == drill_state[1]):
                avg_count = df_grouped[df_grouped['Day'] != current_date]['Redemption Count'].mean()
            else:
                avg_count = df_grouped['Redemption Count'].mean()
            
            # Add bar chart
            fig.add_trace(go.Bar(
                x=df_grouped['Day_Str'],
                y=df_grouped['Redemption Count'],
                name='Redemption Count'
            ))
            
            # Add average line
            fig.add_trace(go.Scatter(
                x=df_grouped['Day_Str'],
                y=[avg_count] * len(df_grouped),
                mode='lines',
                name=f'Average ({avg_count:,.0f})',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(title=drill_titles[2].format(drill_state[1], drill_state[0]))
            
        elif level == 3:
            # Parse the date string from drill_state
            selected_date = datetime.strptime(drill_state[2], "%Y-%m-%d").date()
            df_filtered = df[df['Day'] == selected_date]
            df_grouped = df_filtered.groupby("Hour")['Redemption Count'].sum().reset_index()
            
            # Calculate average excluding current hour if it's today
            current_date = datetime.now().date()
            current_hour = datetime.now().hour
            if selected_date == current_date:
                avg_count = df_grouped[df_grouped['Hour'] != current_hour]['Redemption Count'].mean()
            else:
                avg_count = df_grouped['Redemption Count'].mean()
            
            # Add bar chart
            fig.add_trace(go.Bar(
                x=df_grouped['Hour'],
                y=df_grouped['Redemption Count'],
                name='Redemption Count'
            ))
            
            # Add average line
            fig.add_trace(go.Scatter(
                x=df_grouped['Hour'],
                y=[avg_count] * len(df_grouped),
                mode='lines',
                name=f'Average ({avg_count:,.0f})',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(title=drill_titles[3].format(drill_state[2]))
            
        elif level == 4:
            # Parse the date string from drill_state
            selected_date = datetime.strptime(drill_state[2], "%Y-%m-%d").date()
            df_filtered = df[
                (df['Day'] == selected_date) &
                (df['Hour'] == int(drill_state[3]))
            ]
            df_grouped = df_filtered.groupby("Timestamp")['Redemption Count'].sum().reset_index()
            df_grouped["label"] = df_grouped["Timestamp"].dt.strftime('%H:%M')
            
            # Calculate average excluding current 15-minute period if it's today and current hour
            current_date = datetime.now().date()
            current_hour = datetime.now().hour
            if selected_date == current_date and int(drill_state[3]) == current_hour:
                # Get current 15-minute period
                current_minute = datetime.now().minute
                current_period = current_minute // 15
                current_period_start = current_period * 15
                current_period_end = (current_period + 1) * 15
                
                # Filter out current period
                df_avg = df_grouped[
                    ~((df_grouped["Timestamp"].dt.hour == current_hour) & 
                      (df_grouped["Timestamp"].dt.minute >= current_period_start) & 
                      (df_grouped["Timestamp"].dt.minute < current_period_end))
                ]
                avg_count = df_avg['Redemption Count'].mean()
            else:
                avg_count = df_grouped['Redemption Count'].mean()
            
            # Add bar chart
            fig.add_trace(go.Bar(
                x=df_grouped['label'],
                y=df_grouped['Redemption Count'],
                name='Redemption Count'
            ))
            
            # Add average line
            fig.add_trace(go.Scatter(
                x=df_grouped['label'],
                y=[avg_count] * len(df_grouped),
                mode='lines',
                name=f'Average ({avg_count:,.0f})',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(title=drill_titles[4].format(drill_state[2]))
        else:
            return dash.no_update, session_data
        
        fig.update_xaxes(type='category')
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        return fig, session_data
    except Exception as e:
        print(f"Error updating graph: {e}")
        return px.bar(title="Error loading data"), session_data

# Update the YTD/MTD graph callback
@app.callback(
    Output("ytd-bar-graph", "figure"),
    Output('ytd-session-data', 'data'),
    Input("ytd-bar-graph", "clickData"),
    Input("ytd-back-button", "n_clicks"),
    Input('interval-component', 'n_intervals'),
    State('ytd-session-data', 'data')
)
def update_ytd_graph(clickData, n_clicks, n_intervals, session_data):
    try:
        df = get_processed_data()
        drill_state = session_data.get('drill_state', [])

        ctx = callback_context
        trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        if trigger == "ytd-back-button" and drill_state:
            drill_state.pop()
        elif trigger == "ytd-bar-graph" and clickData is not None and len(drill_state) < 1:
            clicked_value = str(clickData['points'][0]['x'])
            drill_state.append(clicked_value)

        session_data['drill_state'] = drill_state
        level = len(drill_state)

        # Get current date info
        current_date = datetime.now().date()
        current_year = current_date.year
        current_month = current_date.month
        current_day = current_date.day

        # Create figure
        fig = go.Figure()
        
        if level == 0:
            # Get all years and filter for complete years only
            all_years = sorted(df['Year'].unique())
            complete_years = [year for year in all_years if is_complete_year(df, year)]
            ytd_data = []
            
            for year in complete_years:
                year_data = df[df['Year'] == year]
                
                if year == current_year:
                    # For current year, only show up to current date
                    ytd_count = year_data[year_data['Day'] <= current_date]['Redemption Count'].sum()
                else:
                    # For past years, show data up to the same month/day
                    target_date = datetime(year, current_month, current_day).date()
                    ytd_count = year_data[year_data['Day'] <= target_date]['Redemption Count'].sum()
                
                ytd_data.append({'Year': year, 'Count': ytd_count})
            
            # Create DataFrame for plotting
            plot_data = pd.DataFrame(ytd_data)
            
            if not plot_data.empty:
                # Add bars for YTD with text labels
                fig.add_trace(go.Bar(
                    x=plot_data['Year'],
                    y=plot_data['Count'],
                    marker_color='rgb(55, 83, 109)',
                    text=plot_data['Count'].apply(lambda x: f'{x:,.0f}' if x > 0 else '0'),
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title='Year-to-Date (YTD) Comparison',
                    showlegend=False,
                    margin=dict(t=100),
                    xaxis=dict(
                        tickmode='array',
                        ticktext=[str(year) for year in plot_data['Year']],
                        tickvals=plot_data['Year'],
                        tickangle=0,
                        type='category'
                    )
                )
            else:
                fig.update_layout(title='No complete years available for comparison')

        elif level == 1:
            selected_year = int(drill_state[0])
            print(f"\nDEBUG - Processing monthly data for year {selected_year}")
            df_filtered = df[df['Year'] == selected_year]
            
            # Prepare monthly data
            monthly_data = []
            for month in range(1, 13):
                if selected_year == current_year and month > current_month:
                    continue
                
                month_data = df_filtered[df_filtered['Month_Num'] == month]
                if not month_data.empty:
                    if selected_year == current_year and month == current_month:
                        end_date = current_date
                    else:
                        end_date = datetime(selected_year, month, current_day).date()
                    
                    mtd_count = month_data[month_data['Day'] <= end_date]['Redemption Count'].sum()
                    month_name = calendar.month_name[month]
                    print(f"DEBUG - {month_name} {selected_year} count: {mtd_count}")
                    monthly_data.append({'Month': month_name, 'Count': mtd_count})
            
            plot_data = pd.DataFrame(monthly_data)
            print(f"DEBUG - Monthly plot data:\n{plot_data}")
            
            if not plot_data.empty:
                fig.add_trace(go.Bar(
                    x=plot_data['Month'],
                    y=plot_data['Count'],
                    marker_color='rgb(26, 118, 255)',
                    text=plot_data['Count'].apply(lambda x: f'{x:,.0f}'),
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title=f'Monthly Analysis for {selected_year}',
                    showlegend=False,
                    margin=dict(t=100),
                    xaxis=dict(
                        tickmode='array',
                        ticktext=plot_data['Month'],
                        tickvals=plot_data['Month'],
                        tickangle=0
                    )
                )
            else:
                print(f"DEBUG - No monthly data available for {selected_year}")
                fig.update_layout(title=f'No data available for {selected_year}')
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Redemption Count",
            height=500
        )
        
        return fig, session_data
    except Exception as e:
        print(f"Error updating YTD graph: {e}")
        import traceback
        print(traceback.format_exc())
        return px.bar(title="Error loading data"), session_data

if __name__ == '__main__':
    # For local development
    app.run(debug=True, port=8050)
