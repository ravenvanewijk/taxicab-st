import os
import sys
import unittest
import osmnx as ox

# Add the parent directory of 'taxicab' to sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, PARENT_DIR)

import taxicab as ts

NETWORK_PATH = os.path.join(THIS_DIR, 'data/test_graph.osm')

# load graph and add travel times in s
G = ox.load_graphml(NETWORK_PATH)
G = ox.routing.add_edge_speeds(G)
G = ox.routing.add_edge_travel_times(G)


class test_main(unittest.TestCase):
    def test_short_route(self):
        orig = (39.0884, -84.3232)
        dest = (39.08843038088047, -84.32261113356783)
        route = ts.time.shortest_path(G, orig, dest)
        self.assertEqual(route[0], 3.7741818962052203)

    def test_same_edge(self):
        orig = (39.08734, -84.32400)
        dest = (39.08840, -84.32307)
        route = ts.time.shortest_path(G, orig, dest)
        self.assertEqual(route[0], 13.4655474763054)

    def test_far_away_nodes(self):
        orig = (39.08710, -84.31050)
        dest = (39.08800, -84.32000)
        route = ts.time.shortest_path(G, orig, dest)
        self.assertEqual(route[0], 56.59085602066179)

if __name__ == '__main__':
    unittest.main()
