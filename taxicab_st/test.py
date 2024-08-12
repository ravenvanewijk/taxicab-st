
# import osmnx as ox
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from shapely import Point

# def plot_linestring(line, point=None, overlap=True):
#     """Helper function, plots a Shapely LineString

#     args: type, description:
#         line: Shapely Linestring, sequence of coordinates of a geometry
#         point: tuple, coordinates of an additional point to plot
#         overlap: bool, plot on a new figure yes/no"""
#     if not overlap:
#         plt.figure()
#     x, y = line.xy
#     plt.plot(x, y, marker='o')  # Plot the line with markers at vertices
#     plt.plot(x[-1],y[-1],'rs') 
#     if not point is None:
#         plt.plot(point[0], point[1], 'gs')
#     plt.title('LineString Plot')
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.grid(True)


# def plot_graph(G, custs, lines=[]):
#     """Plots the graph of the selected gpkg file as well as customer 
#     locations"""
#     # Plot city graph
#     fig, ax = ox.plot_graph(G, show=False, close=False)
#     # Plot the customers
#     locs_scatter = ax.scatter([point.x for _, point in custs.items()],
#                                     [point.y for _, point in custs.items()],
#                                     c='red', s=30, zorder=10, label='L&R locations')

#     for line in lines:
#         x, y = line.xy
#         ax.plot(x, y, marker='o')  # Plot the line with markers at vertices
#         ax.plot(x[-1],y[-1],'rs') 

#     # Show the plot with a legend
#     ax.legend(handles=[locs_scatter])
#     plt.show()

# def str_interpret(value):
#     return value  # Ensure the value remains a string

# G = ox.load_graphml(filepath='taxicab_st/Buffalo.graphml',
#                         edge_dtypes={'osmid': str_interpret,
#                                     'reversed': str_interpret})

# # custs = pd.Series([Point(-78.81052204466427, 42.909781964425356), Point( -78.811514, 42.909251)])
# custs = pd.Series([Point(-78.73280686053779, 42.996897142139346), Point(-78.733863, 42.997993)])
# t= pd.Series([])

# # plot_graph(G, custs)


# # A = (42.909781964425356, -78.81052204466427)     
# # B = ( 42.909251, -78.811514)

# A = (42.996897142139346, -78.73280686053779)     
# B = np.array([ 42.997993, -78.733863])


# q,w,e,r= shortest_path(G,A,B)    

# plot_linestring(e)
# plot_linestring(r)
# plot_linestring(G.edges[(w[0], w[1], 0)]['geometry'])
# plt.show()



# # BELOW CODE TO CHECK GEOM
# # Define the coordinates for each edge
# coords_orig_partial_edge_2 = ([-78.732801, -78.7328072070703], [42.996894, 42.99689500776838])
# coords_orig_partial_edge_1 = ([-78.7328072070703, -78.7380967], [42.99689500776838, 42.9977538])
# coords_dest_partial_edge_1 = ([-78.73386301951832, -78.7343678], [42.99799286284995, 42.9980647])
# coords_dest_partial_edge_2 = ([-78.732817, -78.73386301951832], [42.997844, 42.99799286284995])

# # Function to convert coordinates to LineString
# def create_line_string(coords):
#     x_coords, y_coords = coords
#     return LineString(zip(x_coords, y_coords))

# # Create LineString objects
# line_orig_partial_edge_2 = create_line_string(coords_orig_partial_edge_2)
# line_orig_partial_edge_1 = create_line_string(coords_orig_partial_edge_1)
# line_dest_partial_edge_1 = create_line_string(coords_dest_partial_edge_1)
# line_dest_partial_edge_2 = create_line_string(coords_dest_partial_edge_2)

# # plot_linestring(line_orig_partial_edge_2)
# # plot_linestring(line_orig_partial_edge_1)
# # plot_linestring(line_dest_partial_edge_2)
# # plot_linestring(line_dest_partial_edge_1)
# # plt.show()


# # plot_graph(G, custs, lines=[line_orig_partial_edge_2, line_orig_partial_edge_1, line_dest_partial_edge_1, line_dest_partial_edge_2])