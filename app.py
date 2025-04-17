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

# Add development mode flag at the top of the file, after imports
DEV_MODE = True  # Set to False in production

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
    try:
        data = load_data()
        
        # Read data in chunks to reduce memory usage
        df = pd.read_json(StringIO(data['ferry_data']), orient='split')
        
        # Process only necessary columns
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Day'] = df['Timestamp'].dt.date
        df['Year'] = df['Timestamp'].dt.year
        df['Month_Num'] = df['Timestamp'].dt.month
        df['Month'] = df['Month_Num'].apply(lambda x: calendar.month_name[x])
        df['Day_Str'] = df['Day'].astype(str)
        df['Hour'] = df['Timestamp'].dt.hour
        
        # Keep only necessary columns
        columns_to_keep = ['Timestamp', 'Day', 'Year', 'Month_Num', 'Month', 'Day_Str', 'Hour', 'Redemption Count']
        df = df[columns_to_keep]
        
        return df
    except Exception as e:
        print(f"Error in get_processed_data: {e}")
        import traceback
        print(traceback.format_exc())
        raise

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
    """Fetch ferry data from the Toronto Open Data API"""
    base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
    url = base_url + "/api/3/action/package_show"
    params = {"id": "toronto-island-ferry-ticket-counts"}
    
    try:
        time.sleep(1)
        print("\nDEBUG - Fetching package info...")
        response = requests.get(url, params=params)
        
        if response.status_code == 429:
            print("DEBUG - Rate limited. Waiting 60 seconds...")
            time.sleep(60)
            response = requests.get(url, params=params)
        
        response.raise_for_status()
        package = response.json()
        
        for idx, resource in enumerate(package["result"]["resources"]):
            if resource["datastore_active"]:
                url = base_url + "/datastore/dump/" + resource["id"]
                print(f"\nDEBUG - Fetching resource {idx} data from:", url)
                time.sleep(1)
                
                response = requests.get(url)
                if response.status_code == 429:
                    print("DEBUG - Rate limited. Waiting 60 seconds...")
                    time.sleep(60)
                    response = requests.get(url)
                
                response.raise_for_status()
                return response.text
        return None
    except requests.exceptions.RequestException as e:
        print(f"DEBUG - Error fetching data: {e}")
        if last_refresh['ferry_data'] is not None:
            print("DEBUG - Using cached data due to API error")
            return last_refresh['ferry_data']
        raise Exception(f"Failed to fetch ferry data: {e}")

def load_data():
    """Load and process data from both API and historical sources"""
    global last_refresh
    print("\nDEBUG - Starting load_data()")
    current_time = datetime.now()
    
    try:
        if DEV_MODE:
            if os.path.exists(HISTORICAL_DATA_FILE):
                print("DEBUG - Dev mode: Using cached data")
                df = pd.read_csv(HISTORICAL_DATA_FILE)
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df['Day'] = pd.to_datetime(df['Timestamp']).dt.date
                last_refresh = {
                    'timestamp': current_time,
                    'ferry_data': df.to_json(date_format='iso', orient='split'),
                    'weather_data': None,
                    'last_backup_date': current_time.date()
                }
                return last_refresh
            else:
                print("DEBUG - Dev mode: No historical data found, fetching from API")
        
        if (last_refresh['timestamp'] is None or 
            current_time - last_refresh['timestamp'] >= timedelta(hours=1)):
            
            print("DEBUG - Fetching new data from API")
            resource_dump_data = fetch_ferry_data()
            if resource_dump_data is None:
                raise Exception("Failed to fetch ferry data")
            
            print("DEBUG - Converting API data to DataFrame")
            df = pd.read_csv(StringIO(resource_dump_data))
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Day'] = df['Timestamp'].dt.date
            
            if not DEV_MODE or not os.path.exists(HISTORICAL_DATA_FILE):
                df = save_to_historical_data(df)
            
            last_refresh = {
                'timestamp': current_time,
                'ferry_data': df.to_json(date_format='iso', orient='split'),
                'weather_data': None,
                'last_backup_date': current_time.date()
            }
            
            if hasattr(get_processed_data, 'cache_clear'):
                get_processed_data.cache_clear()
            
            return last_refresh
        
        return last_refresh
    
    except Exception as e:
        print(f"Error in load_data: {e}")
        import traceback
        print(traceback.format_exc())
        
        if os.path.exists(HISTORICAL_DATA_FILE):
            print("DEBUG - Error occurred, falling back to historical data")
            df = pd.read_csv(HISTORICAL_DATA_FILE)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Day'] = pd.to_datetime(df['Timestamp']).dt.date
            last_refresh = {
                'timestamp': current_time,
                'ferry_data': df.to_json(date_format='iso', orient='split'),
                'weather_data': None,
                'last_backup_date': current_time.date()
            }
            return last_refresh
        raise Exception("No data available")

# App setup
external_stylesheets = [
    dbc.themes.CERULEAN,
    'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Important: This is what Render needs
server = app.server

# Drill-down titles (this is fine as a constant)
drill_titles = [
    'Total Redemptions by Year (Click on a year to see monthly details)',
    'Monthly Redemptions in {} (Click on a month to see daily details)',
    'Daily Redemptions in {} {} (Click on a day to see hourly details)',
    'Hourly Redemptions on {} (Click on an hour to see 15-minute intervals)',
    '15-Minute Redemptions on {}'
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
                    # Add instruction text
                    html.Div([
                        html.I(className="fas fa-info-circle me-2"),
                        "Click on bars to drill down into more detailed views. Use the back button to return to previous views."
                    ], className="text-muted mb-3", style={"fontSize": "0.9rem"}),
                    html.Button("Back", id="back-button", className="btn btn-outline-primary mb-3"),
                    dcc.Loading(
                        id="loading-graph",
                        type="circle",
                        children=[
                            dcc.Graph(
                                id="bar-graph",
                                config={
                                    'responsive': True,
                                    'displayModeBar': True,
                                    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                                    'displaylogo': False
                                },
                                style={'height': '400px'}  # Fixed height
                            )
                        ]
                    )
                ])
            ], className="mb-4 h-100"),  # Added h-100 for full height
            
            # YTD/MTD Analysis Graph
            dbc.Card([
                dbc.CardBody([
                    html.H4("YTD/MTD Analysis", className="card-title"),
                    # Add instruction text
                    html.Div([
                        html.I(className="fas fa-info-circle me-2"),
                        "Click on year bars to see monthly breakdowns."
                    ], className="text-muted mb-3", style={"fontSize": "0.9rem"}),
                    html.Button("Back", id="ytd-back-button", className="btn btn-outline-primary mb-3"),
                    dcc.Loading(
                        id="loading-ytd-graph",
                        type="circle",
                        children=[
                            dcc.Graph(
                                id="ytd-bar-graph",
                                config={
                                    'responsive': True,
                                    'displayModeBar': True,
                                    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                                    'displaylogo': False
                                },
                                style={'height': '400px'}  # Fixed height
                            )
                        ]
                    )
                ])
            ], className="mb-4 h-100")  # Added h-100 for full height
        ], xs=12, md=8, className="d-flex flex-column"),  # Added flex column for vertical alignment
        
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
            ], className="mb-4 g-2"),  # Added g-2 for gap between cards
            
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
            ], className="mb-4 g-2"),  # Added g-2 for gap between cards
            
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
            ], className="g-2")  # Added g-2 for gap between cards
        ], xs=12, md=4, className="d-flex flex-column")  # Added flex column for vertical alignment
    ], className="g-4")  # Added g-4 for gap between columns
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
    """Update the bar graph based on user interactions"""
    ctx = dash.callback_context
    if not ctx.triggered:
        drill_state = 0
        selected_data = []
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'back-button' and n_clicks is not None:
            drill_state = max(0, session_data.get('drill_state', 0) - 1)
            selected_data = session_data.get('selected_data', [])[:-1]
        elif clickData is not None:
            drill_state = session_data.get('drill_state', 0) + 1
            selected_data = session_data.get('selected_data', []) + [clickData['points'][0]['x']]
        else:
            return dash.no_update, dash.no_update

    try:
        df = get_processed_data()
        if df is None or df.empty:
            raise ValueError("No data available")
        
        # Define consistent colors
        colors = {
            'bars': 'rgb(55, 83, 109)',  # Navy blue to match YTD graph
            'line': 'red'                 # Red for the average line
        }

        fig = go.Figure()
        
        if drill_state == 0:  # Year level
            grouped = df.groupby('Year')['Redemption Count'].sum().reset_index()
            yearly_avg = grouped['Redemption Count'].mean()
            
            fig.add_trace(go.Bar(
                x=grouped['Year'],
                y=grouped['Redemption Count'],
                name='Total Redemptions',
                hovertemplate='Year: %{x}<br>Total: %{y:,.0f}<br>Click for monthly details<extra></extra>',
                marker_color=colors['bars']
            ))
            
            fig.add_trace(go.Scatter(
                x=grouped['Year'],
                y=[yearly_avg] * len(grouped),
                mode='lines',
                name='Yearly Average',
                line=dict(color=colors['line'], dash='dot'),
                hovertemplate='Yearly Avg: %{y:,.0f}<extra></extra>'
            ))
            
            title = "Ferry Ticket Redemptions by Year<br><sup>Click on a year to see monthly details</sup>"
        
        elif drill_state == 1:  # Month level
            year = int(selected_data[-1])
            year_data = df[df['Year'] == year]
            grouped = year_data.groupby('Month_Num')['Redemption Count'].sum().reset_index()
            monthly_avg = grouped['Redemption Count'].mean()
            
            month_names = {i: calendar.month_abbr[i] for i in range(1, 13)}
            grouped['Month'] = grouped['Month_Num'].map(month_names)
            
            fig.add_trace(go.Bar(
                x=grouped['Month'],
                y=grouped['Redemption Count'],
                name='Total Redemptions',
                hovertemplate='Month: %{x}<br>Total: %{y:,.0f}<br>Click for daily details<extra></extra>',
                marker_color=colors['bars']
            ))
            
            fig.add_trace(go.Scatter(
                x=grouped['Month'],
                y=[monthly_avg] * len(grouped),
                mode='lines',
                name='Monthly Average',
                line=dict(color=colors['line'], dash='dot'),
                hovertemplate='Monthly Avg: %{y:,.0f}<extra></extra>'
            ))
            
            title = f"Monthly Redemptions for {year}<br><sup>Click on a month to see daily details</sup>"
        
        elif drill_state == 2:  # Day level
            year = int(selected_data[-2])
            month = list(calendar.month_abbr).index(selected_data[-1])
            month_data = df[(df['Year'] == year) & (df['Month_Num'] == month)]
            
            grouped = month_data.groupby('Day')['Redemption Count'].sum().reset_index()
            daily_avg = grouped['Redemption Count'].mean()
            
            fig.add_trace(go.Bar(
                x=grouped['Day'],
                y=grouped['Redemption Count'],
                name='Total Redemptions',
                hovertemplate='Date: %{x|%Y-%m-%d}<br>Total: %{y:,.0f}<br>Click for hourly details<extra></extra>',
                marker_color=colors['bars']
            ))
            
            fig.add_trace(go.Scatter(
                x=grouped['Day'],
                y=[daily_avg] * len(grouped),
                mode='lines',
                name='Daily Average',
                line=dict(color=colors['line'], dash='dot'),
                hovertemplate='Daily Avg: %{y:,.0f}<extra></extra>'
            ))
            
            title = f"Daily Redemptions for {calendar.month_name[month]} {year}<br><sup>Click on a day to see hourly details</sup>"
        
        elif drill_state == 3:  # Hour level
            selected_date = pd.to_datetime(selected_data[-1]).date()
            day_data = df[df['Day'] == selected_date]
            
            grouped = day_data.groupby('Hour')['Redemption Count'].sum().reset_index()
            hourly_avg = grouped['Redemption Count'].mean()
            
            fig.add_trace(go.Bar(
                x=grouped['Hour'],
                y=grouped['Redemption Count'],
                name='Total Redemptions',
                hovertemplate='Hour: %{x}:00<br>Total: %{y:,.0f}<br>Click for 15-min details<extra></extra>',
                marker_color=colors['bars']
            ))
            
            fig.add_trace(go.Scatter(
                x=grouped['Hour'],
                y=[hourly_avg] * len(grouped),
                mode='lines',
                name='Hourly Average',
                line=dict(color=colors['line'], dash='dot'),
                hovertemplate='Hourly Avg: %{y:,.0f}<extra></extra>'
            ))
            
            title = f"Hourly Redemptions for {selected_date}<br><sup>Click on an hour to see 15-minute intervals</sup>"
        
        elif drill_state == 4:  # 15-minute level
            selected_date = pd.to_datetime(selected_data[-2]).date()
            selected_hour = int(selected_data[-1])
            
            # Filter data for the specific hour
            hour_data = df[(df['Day'] == selected_date) & (df['Hour'] == selected_hour)]
            
            # Group by 15-minute intervals
            hour_data['Minute_Group'] = (hour_data['Timestamp'].dt.minute // 15) * 15
            grouped = hour_data.groupby('Minute_Group')['Redemption Count'].sum().reset_index()
            interval_avg = grouped['Redemption Count'].mean()
            
            # Create labels for 15-minute intervals
            grouped['Time_Label'] = grouped['Minute_Group'].apply(lambda x: f"{selected_hour:02d}:{x:02d}")
            
            fig.add_trace(go.Bar(
                x=grouped['Time_Label'],
                y=grouped['Redemption Count'],
                name='Total Redemptions',
                hovertemplate='Time: %{x}<br>Total: %{y:,.0f}<extra></extra>',
                marker_color=colors['bars'],
                text=grouped['Redemption Count'].apply(lambda x: f'{x:,.0f}'),
                textposition='outside'
            ))
            
            fig.add_trace(go.Scatter(
                x=grouped['Time_Label'],
                y=[interval_avg] * len(grouped),
                mode='lines',
                name='15-min Average',
                line=dict(color=colors['line'], dash='dot'),
                hovertemplate='15-min Avg: %{y:,.0f}<extra></extra>'
            ))
            
            title = f"15-Minute Intervals for {selected_date} at {selected_hour:02d}:00"

        # Update layout to match YTD/MTD graph
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                y=0.95,
                yanchor='top',
                font=dict(size=16)
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=80, b=50, l=50, r=50),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            ),
            height=400,
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=False,
                tickangle=45 if drill_state >= 2 else 0,  # Angle labels for better readability in day view and beyond
                tickmode='array',
                ticktext=grouped['Time_Label'] if drill_state == 4 else None,
                tickvals=grouped['Time_Label'] if drill_state == 4 else None,
                type='category'  # Force categorical axis
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGrey',
                title='Redemption Count'
            )
        )
        
        # Add text labels to bars for all levels
        if drill_state == 0:  # Year level
            fig.update_traces(
                text=grouped['Redemption Count'].apply(lambda x: f'{x:,.0f}'),
                textposition='outside',
                selector=dict(type='bar')
            )
        elif drill_state == 1:  # Month level
            fig.update_traces(
                text=grouped['Redemption Count'].apply(lambda x: f'{x:,.0f}'),
                textposition='outside',
                selector=dict(type='bar')
            )
        elif drill_state == 2:  # Day level
            fig.update_traces(
                text=grouped['Redemption Count'].apply(lambda x: f'{x:,.0f}'),
                textposition='outside',
                selector=dict(type='bar')
            )
        elif drill_state == 3:  # Hour level
            fig.update_traces(
                text=grouped['Redemption Count'].apply(lambda x: f'{x:,.0f}'),
                textposition='outside',
                selector=dict(type='bar')
            )
        
        # Update session data
        session_data = {
            'drill_state': drill_state,
            'selected_data': selected_data
        }
        
        return fig, session_data
    
    except Exception as e:
        print(f"Error in update_bar_graph: {str(e)}")
        return dash.no_update, dash.no_update

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

        # Create figure
        fig = go.Figure()
        
        # Define colors to match the first graph
        colors = {
            'level0': 'rgb(55, 83, 109)',   # Navy blue for year level
            'level1': 'rgb(26, 118, 255)'   # Light blue for month level
        }

        if level == 0:
            # Get all years and filter for complete years only
            all_years = sorted(df['Year'].unique())
            complete_years = [year for year in all_years if is_complete_year(df, year)]
            ytd_data = []
            
            for year in complete_years:
                year_data = df[df['Year'] == year]
                
                if year == datetime.now().year:
                    # For current year, only show up to current date
                    ytd_count = year_data[year_data['Day'] <= datetime.now().date()]['Redemption Count'].sum()
                else:
                    # For past years, show data up to the same month/day
                    target_date = datetime(year, datetime.now().month, datetime.now().day).date()
                    ytd_count = year_data[year_data['Day'] <= target_date]['Redemption Count'].sum()
                
                ytd_data.append({'Year': year, 'Count': ytd_count})
            
            # Create DataFrame for plotting
            plot_data = pd.DataFrame(ytd_data)
            
            if not plot_data.empty:
                # Add bars for YTD with text labels
                fig.add_trace(go.Bar(
                    x=plot_data['Year'],
                    y=plot_data['Count'],
                    marker_color=colors['level0'],
                    text=plot_data['Count'].apply(lambda x: f'{x:,.0f}' if x > 0 else '0'),
                    textposition='outside',
                    textfont=dict(size=12),
                    cliponaxis=False
                ))
                
                # Calculate y-axis range to accommodate labels
                max_y = plot_data['Count'].max()
                y_range = [0, max_y * 1.15]
                
                fig.update_layout(
                    title='Year-to-Date (YTD) Comparison',
                    showlegend=False,
                    yaxis=dict(range=y_range),
                    xaxis=dict(
                        tickmode='array',
                        ticktext=[str(year) for year in plot_data['Year']],
                        tickvals=plot_data['Year'],
                        tickangle=0,
                        showgrid=False
                    )
                )
            else:
                fig.update_layout(title='No complete years available for comparison')

        elif level == 1:
            selected_year = int(drill_state[0])
            df_filtered = df[df['Year'] == selected_year]
            
            # Prepare monthly data
            monthly_data = []
            for month in range(1, 13):
                if selected_year == datetime.now().year and month > datetime.now().month:
                    continue
                
                month_data = df_filtered[df_filtered['Month_Num'] == month]
                if not month_data.empty:
                    if selected_year == datetime.now().year and month == datetime.now().month:
                        end_date = datetime.now().date()
                    else:
                        end_date = datetime(selected_year, month, datetime.now().day).date()
                    
                    mtd_count = month_data[month_data['Day'] <= end_date]['Redemption Count'].sum()
                    month_name = calendar.month_name[month]
                    monthly_data.append({'Month': month_name, 'Count': mtd_count})
            
            plot_data = pd.DataFrame(monthly_data)
            
            if not plot_data.empty:
                # Add bars for monthly data with text labels
                fig.add_trace(go.Bar(
                    x=plot_data['Month'],
                    y=plot_data['Count'],
                    marker_color=colors['level1'],
                    text=plot_data['Count'].apply(lambda x: f'{x:,.0f}'),
                    textposition='outside',
                    textfont=dict(size=12),
                    cliponaxis=False
                ))
                
                # Calculate y-axis range to accommodate labels
                max_y = plot_data['Count'].max()
                y_range = [0, max_y * 1.15]
                
                fig.update_layout(
                    title=f'Monthly Analysis for {selected_year}',
                    showlegend=False,
                    yaxis=dict(range=y_range),
                    xaxis=dict(
                        tickmode='array',
                        ticktext=plot_data['Month'],
                        tickvals=plot_data['Month'],
                        tickangle=0,
                        showgrid=False
                    )
                )
            else:
                fig.update_layout(title=f'No data available for {selected_year}')
        
        # Update layout
        fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            ),
            margin=dict(t=80, b=50, l=50, r=50),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.9)"
            ),
            height=400,
            title=dict(
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            title_font=dict(
                size=16
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=False,
                tickangle=0
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGrey',
                title='Redemption Count'
            )
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
