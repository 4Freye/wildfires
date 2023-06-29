from datetime import datetime, timedelta
import plotly.express as px
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
import json
from geopy.point import Point
from geopy.distance import geodesic
import igraph as ig
import pandas as pd
import plotly.graph_objects as go
import networkx as nx


# pre-fire dates
kincaid_pre_date = '20190923'
czu_pre_date = '20200716'
august_pre_date = '20200716'

# start dates
kincaid_start_date = '20191023'
czu_start_date = '20200816'
august_start_date = '20200816'

# contained dates
kincaid_contained_date = '20191106'
czu_contained_date = '20200922'
august_contained_date = '20201111'

# long lats
kincaid_coords = (38.792458, -122.780053)
czu_coords = (37.17162, -122.22275)
august_coords = (39.776, -122.673)

def process_pre_fire_data(df, fire_start_date):
    fire_start_date = datetime.strptime(fire_start_date, '%Y%m%d')
    fire_pre_date = fire_start_date - timedelta(days=30)  # get 30 days prior to the fire_start_date
    fire_start_date = fire_start_date.strftime('%Y%m%d')
    fire_pre_date = fire_pre_date.strftime('%Y%m%d')

    pre_fire_df = df.query('date >= @fire_pre_date and date < @fire_start_date')
    n_days = (pre_fire_df['date'].max() - pre_fire_df['date'].min()).days
    pre_fire_grouped = pre_fire_df.groupby(['geoid_o', 'geoid_d']).agg({'visitor_flows':'sum', 'pop_flows':'sum'})
    
    # normalize by the number of days
    pre_fire_grouped = pre_fire_grouped.multiply(1/n_days)
    pre_fire_grouped.reset_index(inplace=True)
    
    # merge long lat
    pre_fire_merged = pre_fire_grouped.merge(df.drop_duplicates(['geoid_o'])[['geoid_o','lat_o','lng_o']], how='left', on='geoid_o')
    pre_fire_merged = pre_fire_merged.merge(df.drop_duplicates(['geoid_d'])[['geoid_d','lat_d','lng_d']], how='left', on='geoid_d')

    return pre_fire_merged

def process_during_fire_data(df, fire_start_date, fire_contained_date):
    fire_start_date = datetime.strptime(fire_start_date, '%Y%m%d')
    fire_pre_date = fire_start_date - timedelta(days=30)  # get 30 days prior to the fire_start_date
    fire_start_date = fire_start_date.strftime('%Y%m%d')
    fire_pre_date = fire_pre_date.strftime('%Y%m%d')

    during_fire_df = df.query('date >= @fire_start_date and date < @fire_contained_date')
    n_days = (during_fire_df['date'].max() - during_fire_df['date'].min()).days
    during_fire_grouped = during_fire_df.groupby(['geoid_o', 'geoid_d']).agg({'visitor_flows':'sum', 'pop_flows':'sum'})
    
    # normalize by the number of days
    during_fire_grouped = during_fire_grouped.multiply(1/n_days)
    during_fire_grouped.reset_index(inplace=True)
    
    # merge long lat
    during_fire_merged = during_fire_grouped.merge(df.drop_duplicates(['geoid_o'])[['geoid_o','lat_o','lng_o']], how='left', on='geoid_o')
    during_fire_merged = during_fire_merged.merge(df.drop_duplicates(['geoid_d'])[['geoid_d','lat_d','lng_d']], how='left', on='geoid_d')

    return during_fire_merged

def plot_density_map(wildfire_df, fire_name, coords):
    plot = wildfire_df.query('acq_date >= @{}_start_date & acq_date <= @{}_contained_date'.format(fire_name, fire_name))
    fig = px.density_mapbox(plot, lat='latitude', lon='longitude', z='confidence', radius=2,
                            center=dict(lat=coords[0], lon=coords[1]), zoom=7.5,
                            mapbox_style="stamen-terrain",
                            color_continuous_scale='turbo')
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    fig.update_layout(annotations=[
        dict(
            text=fire_name,
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.05,
            showarrow=False,
            font=dict(size=16)
        )
    ])
    
    fig.show()

def lat_long_to_location(lat_longs):
    geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)
    counties = []
    for lat_long in lat_longs:
        _loc = geocode(Point(lat_long))
        if _loc == None:
            counties.append(None)
        else:
            counties.append(_loc.raw['address']['county'])
    return counties

geolocator = Nominatim(user_agent="myapplication")

def create_converter_dict(series, converter, batch = False):
    unique_keys = series.str[0]
    if batch == True:
        unique_vals = converter(series.str[1:])
    else:
        unique_vals = unique_keys.apply(converter)
    mapper_dict = dict(zip(unique_keys, unique_vals))
    return mapper_dict

def get_eigenvector_centrality(pre_fire_df, center_coords, affected_dist):
    # Define the center point
    center = center_coords

    # Calculate the distance between each observation and the center point
    distances_o = pre_fire_df.apply(
        lambda row: geodesic(center, (row['lat_o'], row['lng_o'])).kilometers,
        axis=1
    )

    distances_d = pre_fire_df.apply(
        lambda row: geodesic(center, (row['lat_d'], row['lng_d'])).kilometers,
        axis=1
    )

    # Filter the DataFrame for observations outside the maximum distance
    filtered_data = pre_fire_df[(distances_o <= affected_dist) & (distances_d <= affected_dist)].reset_index(drop=True)
    filtered_data['pop_flows'] = pd.to_numeric(filtered_data['pop_flows'], errors='coerce')

    # Eigenvector centralities for counties
    graph = ig.Graph.TupleList(filtered_data[['geoid_o', 'geoid_d', 'pop_flows']].itertuples(index=False), directed=True, edge_attrs='pop_flows')
    eigen_centralities = graph.eigenvector_centrality(weights='pop_flows')
    centrality_df = pd.concat([pd.Series(graph.vs['name'], name='geoid'), pd.Series(eigen_centralities, name='eigen_centrality')], axis=1)

    return filtered_data, centrality_df

def merge_centrality(filtered_df, centrality_df, county_pop, cali_counties_lat_long_dict):
    # Merge county names
    filtered_df[['county_o', 'county_d']] = filtered_df[['geoid_o', 'geoid_d']].apply(lambda x: x.astype(str).map(cali_counties_lat_long_dict))
    
    # Population estimates
    filtered_df = filtered_df.merge(county_pop, left_on='county_o', right_on='county', how='left').drop('county', axis=1).merge(county_pop, left_on='county_d', right_on='county', how='left', suffixes=('_o', '_d')).drop('county', axis=1)
    
    # Eigenvector centralities
    filtered_df = filtered_df.merge(centrality_df, how='left', left_on='geoid_o', right_on='geoid').drop('geoid', axis=1).merge(centrality_df, how='left', left_on='geoid_d', right_on='geoid', suffixes=('_o', '_d')).drop('geoid', axis=1)
    
    return filtered_df

def plot_centrality(merged_df, fire_name):
    # Create a directed graph
    G = nx.DiGraph()

    # Add edges with weights (eigen centrality values)
    for _, row in merged_df.iterrows():
        G.add_edge(row['county_o'], row['county_d'], weight=row['eigen_centrality_d'])

    # Set node positions based on eigen centrality values
    pos = nx.spring_layout(G)

    # Extract node positions
    node_x = []
    node_y = []
    for node, (x, y) in pos.items():
        node_x.append(x)
        node_y.append(y)

    # Create edges
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.4, color='gray'),
        hoverinfo='none',
        mode='lines')

    # for each edge, get position & size to add to trace
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Create nodes
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=merged_df['county_d'],  # Set node labels
        textposition="bottom center",
        textfont=dict(size=8, color='black'),
        marker=dict(
            showscale=False,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=5,
                title='Eigencentrality of Counties',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # Set node color based on eigen centrality
    node_trace.marker.color = merged_df['eigen_centrality_d']

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Directed Graph of Eigencentrality of Counties, {}'.format(fire_name),
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.005,
                            y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # Show the figure
    fig.show()