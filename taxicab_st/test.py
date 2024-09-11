

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

G = ox.load_graphml(filepath='taxicab_st/Seattle.graphml',
                        edge_dtypes={'osmid': str_interpret,
                                    'reversed': str_interpret})



A = np.array([47.680838, -122.104114])
B = np.array([47.682122, -122.10635])
# B = np.array([47.5665561, -122.3895247])
# A = np.array([47.625187, -122.352789])

# A = np.array([47.608602, -122.285365])
# B = np.array([47.574254, -122.326014])

# A = np.array([47.638784, -122.203969])
# B = np.array([47.656661, -122.30764])


# A = np.array([42.88189546413181, -78.74404160878684])
# B = np.array([42.88198599999998, -78.746419])
# A = np.array([42.87057098882533, -78.7324669405705])
# B = np.array([42.87571, -78.731316])
q,w,e,r,t= shortest_path(G,A,B)
print(w,e,r)

custs = pd.Series([Point(A[1], A[0]), Point(B[1], B[0])])
rte=[]
# nx_route = [53089055, 53089056, 366732331, 53211210, 53211211, 4274337191, 6030052653, 6030052663, 6030018277, 6246050797, 53052457, 2955405514, 53052436, 30830849, 2939959136, 4549961257, 4549961260, 30830863, 2652866671, 2927137481, 5413301435, 11611747359, 7010447307, 2247309277, 4549993731, 4550007326, 59677236, 10166353276, 31429758, 32103268, 799291217, 4684782653, 59713263, 31429756, 32103800, 9417305222, 30079365, 32172259, 59594313, 31251602, 1726065110, 2776431135, 53224599, 59948458, 53079368, 6388490087, 8305962067, 3659288047, 3659288052, 29972814, 9155674446, 3737956368]
for ls in route_to_gdf(G, w)['geometry']:
    rte.append(ls)
# plot_graph(G, custs, [e] + rte)
plot_graph(G, custs, [e,r] + rte)
custs = pd.Series([Point(A[1], A[0]), Point(B[1], B[0])])
rte=[]
nx_route = [8649685293, 53143742]
for ls in route_to_gdf(G, w)['geometry']:
    rte.append(ls)
# plot_graph(G, custs, [e] + rte)
plot_graph(G, custs, [e, r] + rte)