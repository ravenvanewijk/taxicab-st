from networkx import shortest_path as nx_shortest_path

from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.ops import substring

from osmnx.distance import nearest_edges
from osmnx.distance import great_circle
from osmnx.routing import route_to_gdf

from math import dist

def compute_linestring_time(ls, def_spd=30, def_unit='mph'):
    '''
    Computes the length of a partial edge (shapely linesting)
    
    Parameters
    ----------
    ls : shapely.geometry.linestring.LineString

    Returns
    -------
    float : partial edge length distance in meters
    '''
    spd = v2ms(def_spd, def_unit)
    if type(ls) == LineString:
        x, y = zip(*ls.coords)
    
        time = 0
        for i in range(0, len(x)-1):
            time += great_circle(y[i], x[i], y[i+1], x[i+1]) / spd
        return time
    else: return None

def v2ms(value, unit):
    """
    Convert a value given in mph, kph, or kts to meters per second (m/s).

    Args:
    - value (float): The speed value to be converted.
    - unit (str): The unit of the speed value ('mph', 'kph', 'kts').

    Returns:
    - float: The speed in meters per second (m/s).

    Raises:
    - ValueError: If the unit is not one of 'mph', 'kph', 'kts'.
    """
    # Conversion factors from the given unit to meters per second
    conversion_factors = {
        'mph': 0.44704,  
        'kph': 0.27778,
        'kts': 0.51444
    }

    # Check if the provided unit is valid
    if unit not in conversion_factors:
        raise ValueError("Invalid unit. Please use 'mph', 'kph', or 'kts'.")

    # Convert the value to meters per second
    speed_ms = value * conversion_factors[unit]
    return speed_ms

def count_coordinates(geometry):
    return len(list(geometry.coords))

def compute_taxi_time(G, nx_route, orig_partial_edge, dest_partial_edge):
    '''
    Computes the route complete taxi route length
    '''
    timelst = []
    if nx_route:
        gdf = route_to_gdf(G, nx_route)
        # Calculate the number of coordinates and travel time per segment
        # Travel time is assumed linear
        gdf['num_coordinates'] = gdf['geometry'].apply(lambda geom: len(list(geom.coords)) if geom else 2)
        gdf['travel_time_per_segment'] = gdf['travel_time'] / \
                                        (gdf['num_coordinates'] - 1)
        for idx, row in gdf.iterrows():
            # Append the travel time per segment (num_coordinates - 1) times
            timelst.extend([row['travel_time_per_segment']] * \
                            (row['num_coordinates'] - 1))
    if orig_partial_edge:
        timelst = [float(compute_linestring_time(orig_partial_edge))] * \
                    (len(orig_partial_edge.coords) - 1) + timelst
    if dest_partial_edge:
        timelst = timelst + \
                [float(compute_linestring_time(dest_partial_edge))] * \
                    (len(dest_partial_edge.coords) - 1)
    timelst = [0] + timelst
    total_time = sum(timelst)
    return total_time, timelst


def get_edge_geometry(G, edge):
    '''
    Retrieve the points that make up a given edge.
    
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    edge : tuple
        graph edge unique identifier as a tuple: (u, v, key)
    
    Returns
    -------
    list :
        ordered list of (lng, lat) points.
    
    Notes
    -----
    In the event that the 'geometry' key does not exist within the
    OSM graph for the edge in question, it is assumed then that 
    the current edge is just a straight line. This results in an
    automatic assignment of edge end points.
    '''
    
    if G.edges.get(edge, 0):
        if G.edges[edge].get('geometry', 0):
            return G.edges[edge]['geometry']
    
    if G.edges.get((edge[1], edge[0], 0), 0):
        if G.edges[(edge[1], edge[0], 0)].get('geometry', 0):
            return G.edges[(edge[1], edge[0], 0)]['geometry']

    return LineString([
        (G.nodes[edge[0]]['x'], G.nodes[edge[0]]['y']),
        (G.nodes[edge[1]]['x'], G.nodes[edge[1]]['y'])])


def shortest_path(G, orig_yx, dest_yx, orig_edge=None, dest_edge=None):
    '''
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    orig_yx : tuple
        the (lat, lng) or (y, x) point representing the origin of the path
    dest_yx : tuple
        the (lat, lng) or (y, x) point representing the destination of the path
    
    Returns
    -------
    tuple
        (route_dist, route, orig_edge_p, dest_edge_p)
    '''
    
    # determine nearest edges
    if not orig_edge: orig_edge = nearest_edges(G, orig_yx[1], orig_yx[0])
    if not dest_edge: dest_edge = nearest_edges(G, dest_yx[1], dest_yx[0])
    
    # routing along same edge
    if orig_edge == dest_edge:        
        p_o, p_d = Point(orig_yx[::-1]), Point(dest_yx[::-1])
        edge_geo = G.edges[orig_edge]['geometry']
        orig_clip = edge_geo.project(p_o, normalized=True)
        dest_clip = edge_geo.project(p_d, normalized=True)
        orig_partial_edge = substring(edge_geo, orig_clip, dest_clip, 
                                                    normalized=True)  
        dest_partial_edge = []
        nx_route = []
    
    # routing across multiple edges
    else:
        nx_route = nx_shortest_path(G, orig_edge[0], 
                                    dest_edge[0], 'travel_time')
        p_o, p_d = Point(orig_yx[::-1]), Point(dest_yx[::-1])
        orig_geo = get_edge_geometry(G, orig_edge)
        dest_geo = get_edge_geometry(G, dest_edge)

        orig_clip = orig_geo.project(p_o, normalized=True)
        dest_clip = dest_geo.project(p_d, normalized=True)

        orig_partial_edge_1 = substring(orig_geo, orig_clip, 1, normalized=True)
        orig_partial_edge_2 = substring(orig_geo, 0, orig_clip, normalized=True)
        dest_partial_edge_1 = substring(dest_geo, dest_clip, 1, normalized=True)
        dest_partial_edge_2 = substring(dest_geo, 0, dest_clip, normalized=True)   
        # plot_graph(G,lines=[orig_partial_edge_1,orig_partial_edge_2, dest_partial_edge_1,dest_partial_edge_2])
        # If any of these are a line with equal coords, convert to point
        # Will resul in an error if we do not do this     
        try:
            if orig_partial_edge_1.coords[0] == orig_partial_edge_1.coords[1]: 
                orig_partial_edge_1 = Point(orig_partial_edge_1.coords[0])
        except IndexError:
            pass
        try:
            if orig_partial_edge_2.coords[0] == orig_partial_edge_2.coords[1]: 
                orig_partial_edge_2 = Point(orig_partial_edge_2.coords[0])
        except IndexError:
            pass
        try:
            if dest_partial_edge_1.coords[0] == dest_partial_edge_1.coords[1]: 
                dest_partial_edge_1 = Point(dest_partial_edge_1.coords[0])
        except IndexError:
            pass        
        try:
            if dest_partial_edge_2.coords[0] == dest_partial_edge_2.coords[1]: 
                dest_partial_edge_2 = Point(dest_partial_edge_2.coords[0])
        except IndexError:
            pass

        # when the nx route is just a single node, this is a bit of an edge case
        if len(nx_route) <= 2:
            nx_route = []
            if orig_partial_edge_1.intersects(dest_partial_edge_1):
                orig_partial_edge = orig_partial_edge_1
                dest_partial_edge = dest_partial_edge_1
                
            elif orig_partial_edge_1.intersects(dest_partial_edge_2):
                orig_partial_edge = orig_partial_edge_1
                dest_partial_edge = dest_partial_edge_2
                
            elif orig_partial_edge_2.intersects(dest_partial_edge_1):
                orig_partial_edge = orig_partial_edge_2
                dest_partial_edge = dest_partial_edge_1
                
            elif orig_partial_edge_2.intersects(dest_partial_edge_2):
                orig_partial_edge = orig_partial_edge_2
                dest_partial_edge = dest_partial_edge_2
            
            # nx route has apparantly not selected all correct nodes
            # append the route such that there are 3 nodes
            else: 
                nx_route = nx_shortest_path(G, orig_edge[0], 
                                            dest_edge[0], 'travel_time')
                missed_orig_node, orig_node = (orig_edge[0], orig_edge[1]) \
                                    if orig_edge[1] in nx_route else \
                                            (orig_edge[1], orig_edge[0])
                missed_dest_node, dest_node = (dest_edge[0], dest_edge[1]) \
                                    if dest_edge[1] in nx_route else \
                                            (dest_edge[1], dest_edge[0])

                orig_dist_error = dist(orig_yx, (G.nodes[missed_orig_node]['y'],
                                        G.nodes[missed_orig_node]['x'])) - \
                                    dist(orig_yx, (G.nodes[orig_node]['y'],
                                        G.nodes[orig_node]['x']))
                                    
                dest_dist_error = dist(dest_yx, (G.nodes[missed_dest_node]['y'],
                                        G.nodes[missed_dest_node]['x'])) - \
                                    dist(dest_yx, (G.nodes[dest_node]['y'],
                                        G.nodes[dest_node]['x']))

                if orig_dist_error <= dest_dist_error:
                    nx_route = [missed_orig_node] + nx_route

                elif dest_dist_error < orig_dist_error:
                    nx_route = nx_route + [missed_dest_node]            

        # when routing across two or more edges
        if len(nx_route) >= 3:

            ### resolve origin

            # check overlap with first route edge
            route_orig_edge = get_edge_geometry(G, (nx_route[0], 
                                                nx_route[1], 0))
            if orig_edge[0] in nx_route and orig_edge[1] in nx_route:
                nx_route = nx_route[1:]
        
            # determine which origin partial edge to use
            route_orig_edge = get_edge_geometry(G, (nx_route[0], 
                                                nx_route[1], 0)) 
            # We need to check if end or begin point of partial edge matches to
            # beginning or end of original edge
            if orig_partial_edge_1.coords[0] == route_orig_edge.coords[0]\
                or orig_partial_edge_1.coords[-1] == route_orig_edge.coords[0]\
                or orig_partial_edge_1.coords[0] == route_orig_edge.coords[-1]\
                or orig_partial_edge_1.coords[-1] == route_orig_edge.coords[-1]:
                orig_partial_edge = orig_partial_edge_1
            else:
                orig_partial_edge = orig_partial_edge_2

            ### resolve destination

            # check overlap with last route edge
            route_dest_edge = get_edge_geometry(G, (nx_route[-2], 
                                                nx_route[-1], 0))
            if dest_edge[0] in nx_route and dest_edge[1] in nx_route:
                nx_route = nx_route[:-1]

            if len(nx_route) > 1:
                # determine which destination partial edge to use
                route_dest_edge = get_edge_geometry(G, (nx_route[-2], 
                                                    nx_route[-1], 0))
            else:
                nx_route = []
            if dest_partial_edge_1.coords[0] == route_dest_edge.coords[0]\
                or dest_partial_edge_1.coords[-1] == route_dest_edge.coords[0]\
                or dest_partial_edge_1.coords[0] == route_dest_edge.coords[-1]\
                or dest_partial_edge_1.coords[-1] == route_dest_edge.coords[-1]:
                dest_partial_edge = dest_partial_edge_1
            else:
                dest_partial_edge = dest_partial_edge_2
            
    # final check
    try:
        if orig_partial_edge:
            if len(orig_partial_edge.coords) <= 1:
                orig_partial_edge = []
    except UnboundLocalError:
        orig_partial_edge = []

    try:
        if dest_partial_edge:
            if len(dest_partial_edge.coords) <= 1:
                dest_partial_edge = []
    except UnboundLocalError:
        dest_partial_edge = []

    # compute total path length
    route_time, segment_time = compute_taxi_time(G, nx_route, 
                                        orig_partial_edge, dest_partial_edge)

    return route_time, nx_route, orig_partial_edge, \
                                dest_partial_edge, segment_time


import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely import Point


def plot_graph(G, custs=None, lines=[]):
    """Plots the graph of the selected gpkg file as well as customer 
    locations"""
    # Plot city graph
    fig, ax = ox.plot_graph(G, show=False, close=False)
    # Plot the customers
    if custs is not None:
        locs_scatter = ax.scatter([point.x for _, point in custs.items()],
                                        [point.y for _, point in custs.items()],
                                        c='red', s=30, zorder=10, label='L&R locations')
        # Show the plot with a legend
        ax.legend(handles=[locs_scatter])

    for line in lines:
        x, y = line.xy
        ax.plot(x, y, marker='o')  # Plot the line with markers at vertices
        ax.plot(x[-1],y[-1],'rs') 


    plt.show()


def str_interpret(value):
    return value  # Ensure the value remains a string

G = ox.load_graphml(filepath='taxicab_st/Buffalo.graphml',
                        edge_dtypes={'osmid': str_interpret,
                                    'reversed': str_interpret})


B = np.array([42.909906, -78.762629])
A = np.array([42.861058, -78.79152])

# A = np.array([42.948108, -78.762627])
# B = np.array([42.894466, -78.717194])


q,w,e,r,t= shortest_path(G,A,B)
print(w,e,r)

custs = pd.Series([Point(A[1], A[0]), Point(B[1], B[0])])
rte=[]
nx_route = [111449408, 293625412, 264355109, 264348007, 264353727, 8873551940, 1014110131, 111355100, 443517911, 443517905, 111355080, 443517409, 111355062, 111355060, 111355056, 111355044, 111355042, 111355039, 111355035, 111355033, 111355010, 111355008, 111355003, 111354999, 111354996, 111354978, 111354949, 111354944, 111320785, 111303760, 111320783, 111320781, 111320779, 111320777, 111320775, 111320773, 111320771, 111320769, 111320768, 111320757, 6294556286, 111348878, 111348849, 111520862]
for ls in route_to_gdf(G, w)['geometry']:
    rte.append(ls)
# plot_graph(G, custs, [e] + rte)
plot_graph(G, custs, [e, r] + rte)