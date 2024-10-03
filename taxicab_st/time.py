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

def edges_match(edge1, edge2):
    return (edge1.coords[0] == edge2.coords[0] or
            edge1.coords[0] == edge2.coords[-1] or
            edge1.coords[-1] == edge2.coords[0] or
            edge1.coords[-1] == edge2.coords[-1])

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
        gdf['num_coordinates'] = gdf['geometry'].apply(lambda geom: 
                                        len(list(geom.coords)) if geom else 2)
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
        # Very rarely a different orig edge or final edge is chosen by taxicab
        # Than the one selected by nx
        # We want to adjust this here if this is the case
        # this only occurs if its also the final part of the route
        if len(nx_route) >= 2 and orig_edge[0] == nx_route[0] and \
            orig_edge[1] != nx_route[1] and orig_edge[1] in nx_route and \
            orig_yx[0] == G.nodes[nx_route[0]]['y'] and \
            orig_yx[1] == G.nodes[nx_route[0]]['x']:
            orig_edge = tuple(map(int, route_to_gdf(G, nx_route[:2]).index[0]))
        
        if len(nx_route) >= 2 and dest_edge[0] == nx_route[-1] and \
            dest_edge[1] != nx_route[-2] and dest_edge[1] in nx_route and \
            dest_yx[0] == G.nodes[nx_route[-1]]['y'] and \
            dest_yx[1] == G.nodes[nx_route[-1]]['x']:
            dest_edge = tuple(map(int, route_to_gdf(G, nx_route[-2:]).index[0]))
        
        if (nx_route[0] in {dest_edge[0], dest_edge[1], 
                                            orig_edge[0], orig_edge[1]} and
                            nx_route[-1] in {dest_edge[0], dest_edge[1], 
                                            orig_edge[0], orig_edge[1]} and
            len({orig_edge[0], orig_edge[1]}.intersection({dest_edge[0], 
                                                        dest_edge[1]})) > 0):
            # Begin/ final route is sufficient, bypass nx routing
            nx_route = []

        p_o, p_d = Point(orig_yx[::-1]), Point(dest_yx[::-1])
        orig_geo = get_edge_geometry(G, orig_edge)
        dest_geo = get_edge_geometry(G, dest_edge)

        orig_clip = orig_geo.project(p_o, normalized=True)
        dest_clip = dest_geo.project(p_d, normalized=True)

        orig_partial_edge_1 = substring(orig_geo, orig_clip, 1, normalized=True)
        orig_partial_edge_2 = substring(orig_geo, 0, orig_clip, normalized=True)
        dest_partial_edge_1 = substring(dest_geo, dest_clip, 1, normalized=True)
        dest_partial_edge_2 = substring(dest_geo, 0, dest_clip, normalized=True)   
        # If any of these are a line with equal coords, convert to point
        # Will resul in an error if we do not do this     
        try:
            if orig_partial_edge_1.coords[0] == orig_partial_edge_1.coords[-1] and \
                len(orig_partial_edge_1.coords) == 2: 
                orig_partial_edge_1 = Point(orig_partial_edge_1.coords[0])
        except IndexError:
            pass
        try:
            if orig_partial_edge_2.coords[0] == orig_partial_edge_2.coords[-1] and \
                len(orig_partial_edge_1.coords) == 2: 
                orig_partial_edge_2 = Point(orig_partial_edge_2.coords[0])
        except IndexError:
            pass
        try:
            if dest_partial_edge_1.coords[0] == dest_partial_edge_1.coords[-1] and \
                len(orig_partial_edge_1.coords) == 2: 
                dest_partial_edge_1 = Point(dest_partial_edge_1.coords[0])
        except IndexError:
            pass        
        try:
            if dest_partial_edge_2.coords[0] == dest_partial_edge_2.coords[-1] and \
                len(orig_partial_edge_1.coords) == 2: 
                dest_partial_edge_2 = Point(dest_partial_edge_2.coords[0])
        except IndexError:
            pass

        # when the nx route is just a single node, this is a bit of an edge case
        if len(nx_route) <= 2:
            nx_route = []
            if edges_match(orig_partial_edge_1, dest_partial_edge_1):
                orig_partial_edge = orig_partial_edge_1
                dest_partial_edge = dest_partial_edge_1

            elif edges_match(orig_partial_edge_1, dest_partial_edge_2):
                orig_partial_edge = orig_partial_edge_1
                dest_partial_edge = dest_partial_edge_2

            elif edges_match(orig_partial_edge_2, dest_partial_edge_1):
                orig_partial_edge = orig_partial_edge_2
                dest_partial_edge = dest_partial_edge_1

            elif edges_match(orig_partial_edge_2, dest_partial_edge_2):
                orig_partial_edge = orig_partial_edge_2
                dest_partial_edge = dest_partial_edge_2
            
            # nx route has apparently not selected all correct nodes
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

                if abs(orig_dist_error) <= abs(dest_dist_error):
                    nx_route = [missed_orig_node] + nx_route

                elif abs(dest_dist_error) < abs(orig_dist_error):
                    nx_route = nx_route + [missed_dest_node]   

        # when routing across two or more edges
        if len(nx_route) >= 3:

            ### resolve origin

            # check overlap with first route edge
            route_orig_edge = get_edge_geometry(G, (nx_route[0], 
                                                nx_route[1], 0))
            
            # Check whether the edges are connected. 
            # if we previously added an edge it might not be
            # this is because of one way streets.
            # NOT IMPLEMENTED RIGHT NOW AND TODO
            if (nx_route[0], nx_route[1], 0) in G.edges or \
                    (nx_route[0], nx_route[1], 1) in G.edges and \
                (nx_route[-2], nx_route[-1], 0) in G.edges or \
                    (nx_route[-2], nx_route[-1], 1) in G.edges:
                nx_edges = list(route_to_gdf(G, nx_route).index)
            elif (nx_route[0], nx_route[1], 0) not in G.edges and \
                    (nx_route[0], nx_route[1], 1) not in G.edges:
                nx_edges = list(route_to_gdf(G, [nx_route[1]] + \
                                [nx_route[0]]).index) + \
                        list(route_to_gdf(G, nx_route[1:]).index)
            else:
                nx_edges = list(route_to_gdf(G, nx_route[:-1]).index) + \
                    list(route_to_gdf(G, [nx_route[-1]] + \
                                [nx_route[-2]]).index)

            nx_edges_uv = {(u, v) for u, v, key in nx_edges}

            # Remove the first node from nx route if it is already encapsulated
            # in the final linepiece. We check this by checking what edge the 
            # first linepiece is made of
            # Exception is when it is circular. In that case we want to check
            # for the second node as well, in case that is also the same
            orig_edge_in_nx = (orig_edge[0], orig_edge[1]) in nx_edges_uv or \
                                (orig_edge[1], orig_edge[0]) in nx_edges_uv
            is_not_self_loop = orig_edge[0] != orig_edge[1]
            is_start_self_loop = orig_edge[0] == orig_edge[1] == \
                nx_route[0] == nx_route[1]

            # Combine conditions
            if (orig_edge_in_nx and is_not_self_loop) or is_start_self_loop:
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
                # Additional check: if its just a point and it matches, we want
                # the other partial edge.
                if type(orig_partial_edge_2) == Point and \
                    (orig_partial_edge_2.coords[0] == \
                                            route_orig_edge.coords[0]\
                    or orig_partial_edge_2.coords[0] == \
                                            route_orig_edge.coords[-1]):
                    orig_partial_edge = orig_partial_edge_2
                else:
                    orig_partial_edge = orig_partial_edge_1
            else:
                orig_partial_edge = orig_partial_edge_2

            ### resolve destination

            # check overlap with last route edge
            route_dest_edge = get_edge_geometry(G, (nx_route[-2], 
                                                nx_route[-1], 0))
            # Remove the last node from nx route if it is already encapsulated
            # in the final linepiece. We check this by checking what edge the 
            # last linepiece is made of
            # Exception is when it is circular. In that case we want to check
            # for the second to last node as well, in case that's also the same
            nx_edges = list(route_to_gdf(G, nx_route).index)
            nx_edges_uv = {(u, v) for u, v, key in nx_edges}
            
            dest_edge_in_nx = (dest_edge[0], dest_edge[1]) in nx_edges_uv or \
                                (dest_edge[1], dest_edge[0]) in nx_edges_uv
            is_not_self_loop = dest_edge[0] != dest_edge[1]
            is_end_self_loop = dest_edge[0] == dest_edge[1] == \
                nx_route[-1] == nx_route[-2]

            # Combine conditions
            if (dest_edge_in_nx and is_not_self_loop) or is_end_self_loop:
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
                if type(dest_partial_edge_2) == Point and \
                    (dest_partial_edge_2.coords[0] == \
                                            route_dest_edge.coords[0]\
                    or dest_partial_edge_2.coords[0] == \
                                            route_dest_edge.coords[-1]):
                    dest_partial_edge = dest_partial_edge_2
                else:
                    dest_partial_edge = dest_partial_edge_1
            else:
                dest_partial_edge = dest_partial_edge_2
            
    # final check
    try:
        if orig_partial_edge:
            if len(orig_partial_edge.coords) <= 1:
                orig_partial_edge = []
            elif len(orig_partial_edge.coords) == 2 and \
                orig_partial_edge.coords[0] == orig_partial_edge.coords[-1]:
                orig_partial_edge = []
    except UnboundLocalError:
        orig_partial_edge = []

    try:
        if dest_partial_edge:
            if len(dest_partial_edge.coords) <= 1:
                dest_partial_edge = []
            elif len(dest_partial_edge.coords) == 2 and \
                dest_partial_edge.coords[0] == dest_partial_edge.coords[-1]:
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


# A = np.array([47.687109, -122.120318])
# B = np.array([47.680838, -122.104114])


# A = np.array([42.875181199999986, -78.861864])
# B = np.array([ 42.856027, -78.867927])

# A = np.array([42.92238771355551, -78.83363366913012])
# B = np.array([42.92179680000001, -78.8336239])
# A = np.array([42.961694872237025, -78.7593302452336])
# B = np.array([ 42.965185599999984, -78.7593501])

# A = np.array([42.94089282415961, -78.82162294637753])
# B = np.array([ 42.94009299999999, -78.822945])

# A = np.array([42.998927324626194, -78.81076762202092])
# B = np.array([ 43.000319700000006, -78.8119001])
# A = np.array([42.914356, -78.796431])
# B = np.array([ 42.912856, -78.821322])


# A = np.array([42.99148177004806, -78.77108391446286])
# B = np.array([ 42.99134759999998, -78.7821653])

# A = np.array([42.8876712, -78.7677336])
# B = np.array([ 42.86671 , -78.801124])
# A = np.array([42.908696, -78.766809])
# B = np.array([ 42.907394, -78.751083])

# A = np.array([47.547134, -122.336966])
# B = np.array([47.538336, -122.295355])

# A = np.array([47.680838, -122.104114])
# B = np.array([47.682122, -122.10635])
# B = np.array([47.5665561, -122.3895247])
# A = np.array([47.625187, -122.352789])

# A = np.array([47.608602, -122.285365])
# B = np.array([47.574254, -122.326014])

# A = np.array([47.638784, -122.203969])
# B = np.array([47.656661, -122.30764])

# below is error due to removing edge unnecessarily
# A = np.array([ 42.995467, -78.81462])
# B = np.array([42.98816 , -78.822566])

# below gives not connected error 2x
A = np.array([ 42.959665588770186, -78.76380033011569])
B = np.array([42.959674699999994, -78.7635801])
# A = np.array([ 42.92011654552182, -78.89254777293446])
# B = np.array([42.9201281, -78.8916306])

# A = np.array([42.8876712, -78.7677336])
# B = np.array([ 42.86671 , -78.801124])
# # A = np.array([42.88189546413181, -78.74404160878684])
# # B = np.array([42.88198599999998, -78.746419])
# # A = np.array([42.87057098882533, -78.7324669405705])
# # B = np.array([42.87571, -78.731316])
q,w,e,r,t= shortest_path(G,A,B)
print(w,e,r)
# # A = (G.nodes[111440824]['y'], G.nodes[111440824]['x'])
# # B = (G.nodes[111390208]['y'], G.nodes[111390208]['x'])
custs = pd.Series([Point(A[1], A[0]), Point(B[1], B[0])])
rte=[]
# for ls in route_to_gdf(G, w)['geometry']:
#     rte.append(ls)
plot_graph(G, custs , [e]+ rte )

# nx_route = [1214320545, 301022640, 111581657, 111275949, 111303050, 1214320684, 301025662, 264585683, 301025695, 301025723]
# rte= []
# for ls in route_to_gdf(G, nx_route)['geometry']:
#     rte.append(ls)
# plot_graph(G, custs , rte )