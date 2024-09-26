

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

A = np.array([42.998927324626194, -78.81076762202092])
B = np.array([ 43.000319700000006, -78.8119001])


# A = np.array([42.99148177004806, -78.77108391446286])
# B = np.array([ 42.99134759999998, -78.7821653])

# A = np.array([42.8876712, -78.7677336])
# B = np.array([ 42.86671 , -78.801124])


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


# A = np.array([42.88189546413181, -78.74404160878684])
# B = np.array([42.88198599999998, -78.746419])
# A = np.array([42.87057098882533, -78.7324669405705])
# B = np.array([42.87571, -78.731316])
q,w,e,r,t= shortest_path(G,A,B)
print(w,e,r)
# A = (G.nodes[111440824]['y'], G.nodes[111440824]['x'])
# B = (G.nodes[111390208]['y'], G.nodes[111390208]['x'])
custs = pd.Series([Point(A[1], A[0]), Point(B[1], B[0])])
rte=[]
nx_route = [273227728]
for ls in route_to_gdf(G, w)['geometry']:
    rte.append(ls)
plot_graph(G, custs , [e, r]+ rte )