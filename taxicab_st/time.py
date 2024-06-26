from networkx import shortest_path as nx_shortest_path

from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.ops import substring

from osmnx.distance import nearest_edges
from osmnx.distance import great_circle
from osmnx.routing import route_to_gdf


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
        'mph': 0.44704,  # 1 mile per hour is approximately 0.44704 meters per second
        'kph': 0.27778,  # 1 kilometer per hour is approximately 0.27778 meters per second
        'kts': 0.51444   # 1 knot is approximately 0.51444 meters per second
    }

    # Check if the provided unit is valid
    if unit not in conversion_factors:
        raise ValueError("Invalid unit. Please use 'mph', 'kph', or 'kts'.")

    # Convert the value to meters per second
    speed_ms = value * conversion_factors[unit]
    return speed_ms


def compute_taxi_time(G, nx_route, orig_partial_edge, dest_partial_edge):
    '''
    Computes the route complete taxi route length
    '''

    time = 0
    if nx_route:
        time += sum(route_to_gdf(G, nx_route)['travel_time'])
    if orig_partial_edge:
        time += compute_linestring_time(orig_partial_edge)
    if dest_partial_edge:
        time += compute_linestring_time(dest_partial_edge)
    return time


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
        orig_partial_edge = substring(edge_geo, orig_clip, dest_clip, normalized=True)  
        dest_partial_edge = []
        nx_route = []
    
    # routing across multiple edges
    else:
        nx_route = nx_shortest_path(G, orig_edge[0], dest_edge[0], 'travel_time')
        p_o, p_d = Point(orig_yx[::-1]), Point(dest_yx[::-1])
        orig_geo = get_edge_geometry(G, orig_edge)
        dest_geo = get_edge_geometry(G, dest_edge)

        orig_clip = orig_geo.project(p_o, normalized=True)
        dest_clip = dest_geo.project(p_d, normalized=True)

        orig_partial_edge_1 = substring(orig_geo, orig_clip, 1, normalized=True)
        orig_partial_edge_2 = substring(orig_geo, 0, orig_clip, normalized=True)
        dest_partial_edge_1 = substring(dest_geo, dest_clip, 1, normalized=True)
        dest_partial_edge_2 = substring(dest_geo, 0, dest_clip, normalized=True)        
        
        # when the nx route is just a single node, this is a bit of an edge case
        if len(nx_route) <= 2:
            nx_route = []
            if orig_partial_edge_1.intersects(dest_partial_edge_1):
                orig_partial_edge = orig_partial_edge_1
                dest_partial_edge = dest_partial_edge_1
                
            if orig_partial_edge_1.intersects(dest_partial_edge_2):
                orig_partial_edge = orig_partial_edge_1
                dest_partial_edge = dest_partial_edge_2
                
            if orig_partial_edge_2.intersects(dest_partial_edge_1):
                orig_partial_edge = orig_partial_edge_2
                dest_partial_edge = dest_partial_edge_1
                
            if orig_partial_edge_2.intersects(dest_partial_edge_2):
                orig_partial_edge = orig_partial_edge_2
                dest_partial_edge = dest_partial_edge_2
            
        # when routing across two or more edges
        if len(nx_route) >= 3:

            ### resolve origin

            # check overlap with first route edge
            route_orig_edge = get_edge_geometry(G, (nx_route[0], nx_route[1], 0))
            if route_orig_edge.intersects(orig_partial_edge_1) and route_orig_edge.intersects(orig_partial_edge_2):
                nx_route = nx_route[1:]
        
            # determine which origin partial edge to use
            route_orig_edge = get_edge_geometry(G, (nx_route[0], nx_route[1], 0)) 
            if route_orig_edge.intersects(orig_partial_edge_1):
                orig_partial_edge = orig_partial_edge_1
            else:
                orig_partial_edge = orig_partial_edge_2

            ### resolve destination

            # check overlap with last route edge
            route_dest_edge = get_edge_geometry(G, (nx_route[-2], nx_route[-1], 0))
            if route_dest_edge.intersects(dest_partial_edge_1) and route_dest_edge.intersects(dest_partial_edge_2):
                nx_route = nx_route[:-1]

            # determine which destination partial edge to use
            route_dest_edge = get_edge_geometry(G, (nx_route[-2], nx_route[-1], 0)) 
            if route_dest_edge.intersects(dest_partial_edge_1):
                dest_partial_edge = dest_partial_edge_1
            else:
                dest_partial_edge = dest_partial_edge_2
            
    # final check
    if orig_partial_edge:
        if len(orig_partial_edge.coords) <= 1:
            orig_partial_edge = []
    if dest_partial_edge:
        if len(dest_partial_edge.coords) <= 1:
            dest_partial_edge = []

    # compute total path length
    route_dist = compute_taxi_time(G, nx_route, orig_partial_edge, dest_partial_edge)

    return route_dist, nx_route, orig_partial_edge, dest_partial_edge
