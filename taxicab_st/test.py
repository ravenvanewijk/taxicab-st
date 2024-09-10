

import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely import Point

def plot_linestring(line, point=None, overlap=True):
    """Helper function, plots a Shapely LineString

    args: type, description:
        line: Shapely Linestring, sequence of coordinates of a geometry
        point: tuple, coordinates of an additional point to plot
        overlap: bool, plot on a new figure yes/no"""
    if not overlap:
        plt.figure()
    x, y = line.xy
    plt.plot(x, y, marker='o')  # Plot the line with markers at vertices
    plt.plot(x[-1],y[-1],'rs') 
    if not point is None:
        plt.plot(point[0], point[1], 'gs')
    plt.title('LineString Plot')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)

def plot_graph(G, custs, lines=[]):
    """Plots the graph of the selected gpkg file as well as customer 
    locations"""
    # Plot city graph
    fig, ax = ox.plot_graph(G, show=False, close=False)
    # Plot the customers
    locs_scatter = ax.scatter([point.x for _, point in custs.items()],
                                    [point.y for _, point in custs.items()],
                                    c='red', s=30, zorder=10, label='L&R locations')

    for line in lines:
        x, y = line.xy
        ax.plot(x, y, marker='o')  # Plot the line with markers at vertices
        ax.plot(x[-1],y[-1],'rs') 

    # Show the plot with a legend
    ax.legend(handles=[locs_scatter])
    plt.show()

def str_interpret(value):
    return value  # Ensure the value remains a string

G = ox.load_graphml(filepath='taxicab_st/Buffalo.graphml',
                        edge_dtypes={'osmid': str_interpret,
                                    'reversed': str_interpret})

A = (G.nodes[7779745399]['y'], G.nodes[7779745399]['x'])
B = (G.nodes[820000923]['y'], G.nodes[820000923]['x'])


A = np.array([42.87057098882533, -78.7324669405705])
B = np.array([42.87571, -78.731316])
# 1052605684

# A = np.array([42.876466914460224, -78.78590820757644])
# B = np.array([42.868358900000004, -78.8312416])

# A = np.array([42.88189546413181, -78.74404160878684])
# B = np.array([42.88198599999998, -78.746419])

q,w,e,r,t= shortest_path(G,A,B)
print(e)
print(r)
# [226769764, 64132148, 229520889, 69673744, 364051851, 364051942, 364052443]
custs = pd.Series([Point(A[1], A[0]), Point(B[1], B[0])])
rte=[]
for ls in route_to_gdf(G, w)['geometry']:
    rte.append(ls)
# plot_graph(G, custs, [e] + rte)
plot_graph(G, custs, [e, r] + rte)




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


A = np.array([42.948108, -78.762627])
B = np.array([42.894466, -78.717194])

# A = np.array([42.88189546413181, -78.74404160878684])
# B = np.array([42.88198599999998, -78.746419])
A = np.array([42.87057098882533, -78.7324669405705])
B = np.array([42.87571, -78.731316])
q,w,e,r,t= shortest_path(G,A,B)
print(w,e,r)

custs = pd.Series([Point(A[1], A[0]), Point(B[1], B[0])])
rte=[]
nx_route = [111449408, 293625412, 264355109, 264348007, 264353727, 8873551940, 1014110131, 111355100, 443517911, 443517905, 111355080, 443517409, 111355062, 111355060, 111355056, 111355044, 111355042, 111355039, 111355035, 111355033, 111355010, 111355008, 111355003, 111354999, 111354996, 111354978, 111354949, 111354944, 111320785, 111303760, 111320783, 111320781, 111320779, 111320777, 111320775, 111320773, 111320771, 111320769, 111320768, 111320757, 6294556286, 111348878, 111348849, 111520862]
for ls in route_to_gdf(G, w)['geometry']:
    rte.append(ls)
# plot_graph(G, custs, [e] + rte)
plot_graph(G, custs, [e, r] + rte)