from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from geopy.point import Point
from geopy.distance import geodesic
import igraph as ig
import networkx as nx
from shapely.geometry import Point, LineString, shape
from shapely.geometry import MultiPoint, MultiPolygon
from haversine import haversine, Unit
import pyarrow
from utils import *
import json



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


def mark_blocked_paths(df_fire, df_travel, save_path, buffer_radius_km):

    # GeoDataFrame for travel data
    geometry_travel = [LineString([(row['lng_o'], row['lat_o']), (row['lng_d'], row['lat_d'])]) for idx, row in df_travel.iterrows()]
    gdf_travel = gpd.GeoDataFrame(df_travel, geometry=geometry_travel)

    # Set the CRS for the GeoDataFrames
    gdf_travel.set_crs("EPSG:4326", inplace=True)

    # For each unique date in the fire data, create a Polygon (buffered Convex Hull) that encompasses all points for that date
    df_fire['acq_date'] = pd.to_datetime(df_fire['acq_date'])
    unique_dates = df_fire['acq_date'].dt.date.unique()
    fire_polygons = []
    for date in unique_dates:
        df_date = df_fire[df_fire['acq_date'].dt.date == date]
        fire_points = MultiPoint([xy for xy in zip(df_date['longitude'], df_date['latitude'])])
        fire_polygon = gpd.GeoSeries(fire_points.convex_hull, crs="EPSG:4326")  # Create GeoSeries

        # Buffer the polygon
        # First, project the GeoDataFrame to a coordinate system where the unit is meters (e.g., UTM)
        fire_polygon_utm = fire_polygon.to_crs('EPSG:32610')  # Replace 'EPSG:32610' with the correct UTM zone if needed
        buffered_fire_polygon_utm = fire_polygon_utm.buffer(buffer_radius_km * 1000)  # Buffer the polygon in the projected coordinate system

        # Then, project the GeoDataFrame back to WGS84 (latitude/longitude coordinates)
        buffered_fire_polygon = buffered_fire_polygon_utm.to_crs('EPSG:4326')

        fire_polygons.append(buffered_fire_polygon.iloc[0])

    # GeoDataFrame for fire data
    gdf_fire = gpd.GeoDataFrame(df_fire['acq_date'].drop_duplicates().reset_index(drop=True), geometry=fire_polygons)
    gdf_fire.set_crs("EPSG:4326", inplace=True)

    # Save fire polygons as GeoJSON
    gdf_fire.to_file(save_path, driver="GeoJSON")

    join_result = gpd.sjoin(gdf_travel, gdf_fire, how="left", op="intersects")

    blocked_paths = join_result[join_result.index_right.notnull()].index.unique()

    df_travel['blocked'] = False
    for path in blocked_paths:
        df_travel.loc[path, 'blocked'] = True

    # Create a DataFrame that pairs counties
    df_pairs = df_travel[['geoid_o', 'geoid_d']].drop_duplicates().sort_values(by=['geoid_o', 'geoid_d']).reset_index(drop=True)
    df_pairs['blocked'] = False  # Initially mark all pairs as not blocked

    # For each blocked path, mark the corresponding pair of counties as blocked
    for path in blocked_paths:
        geoid_o = df_travel.loc[path, 'geoid_o']
        geoid_d = df_travel.loc[path, 'geoid_d']
        df_pairs.loc[(df_pairs['geoid_o'] == geoid_o) & (df_pairs['geoid_d'] == geoid_d), 'blocked'] = True

    return df_pairs

def plot_data(df_fire, df_travel, df_pairs, fire_name):
    # Convert to GeoDataFrames
    geometry_fire = [Point(xy) for xy in zip(df_fire.longitude, df_fire.latitude)]
    gdf_fire = gpd.GeoDataFrame(df_fire, geometry=geometry_fire)

    geometry_travel = [LineString([(row['lng_o'], row['lat_o']), (row['lng_d'], row['lat_d'])]) for idx, row in df_travel.iterrows()]
    gdf_travel = gpd.GeoDataFrame(df_travel, geometry=geometry_travel)
    
    # Merge df_pairs with df_travel to bring back the geometry of the travel paths
    df_pairs_with_geometry = pd.merge(df_pairs, df_travel[['geoid_o', 'geoid_d', 'geometry']], on=['geoid_o', 'geoid_d'], how='left')
    
    df_pairs_blocked = df_pairs_with_geometry[df_pairs_with_geometry['blocked'] == True]
    gdf_travel_blocked = gpd.GeoDataFrame(df_pairs_blocked, geometry=df_pairs_blocked['geometry'])

    # Create plot
    fig, ax = plt.subplots(figsize=(15, 15))
    gdf_travel.plot(ax=ax, color='blue', label='Travel routes', alpha = 0.02)
    gdf_travel_blocked.plot(ax=ax, color='yellow', label='Blocked routes', alpha=0.5)
    gdf_fire.plot(ax=ax, color='red', markersize=100, label='Fire location', alpha=1.0)
    plt.title(f'Fire and blocked paths for {fire_name}')
    plt.legend()
    plt.show()

def process_geojson(file_path):
    with open(file_path, 'r') as file:
        geojson = json.load(file)
    
    dates = []
    areas = []

    for feature in geojson['features']:
        # Extract the acquisition date and convert it to a datetime object
        date_str = feature['properties']['acq_date']
        date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
        dates.append(date)

        # Calculate the area of the polygon and convert it to square kilometers
        geom = shape(feature['geometry'])
        area = geom.area / 1_000_000  # Convert from square meters to square kilometers
        areas.append(area)
    
    return dates, areas