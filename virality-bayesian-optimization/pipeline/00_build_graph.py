"""
00_build_graph.py
Build the retweet diffusion graph used by all BO experiments.

Input : higgs-retweet_network.edgelist.gz  (SNAP Higgs dataset)
Output: rt_graph.pkl  (largest weakly-connected component)

Download the source file from:
  https://snap.stanford.edu/data/higgs-twitter.html
  (file: higgs-retweet_network.edgelist.gz)
Place it next to this script, or pass its path as argv[1].

EDGE DIRECTION (fixed Nov 2026): each SNAP line "A B w" means A retweeted B,
so information flows B -> A. Earlier versions of this script added A -> B,
which made every cascade run from the retweeter to the account it retweeted.
We now add B -> A, so a node's out-degree is the number of accounts that
retweeted it, and IC propagates along successors in the diffusion direction.
"""
import os, sys, gzip, pickle
import networkx as nx
from common import DIRECTION, GRAPH_PKL

here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, 'higgs-retweet_network.edgelist.gz')

G = nx.DiGraph()
with gzip.open(src, 'rt') as f:
    for line in f:
        retweeter, retweeted, w = line.split()
        # influence flows retweeted -> retweeter
        G.add_edge(int(retweeted), int(retweeter), weight=int(w))

print('retweet graph: nodes=%d edges=%d' % (G.number_of_nodes(), G.number_of_edges()))

# largest weakly-connected component = clean diffusion substrate
wcc = max(nx.weakly_connected_components(G), key=len)
H = G.subgraph(wcc).copy()
H.graph['direction'] = DIRECTION
print('largest WCC: nodes=%d edges=%d' % (H.number_of_nodes(), H.number_of_edges()))

# sanity check: the most-retweeted account should now have the largest out-degree
top = max(H.nodes, key=H.out_degree)
print('max out-degree node %d: out=%d (retweeters), in=%d' % (top, H.out_degree(top), H.in_degree(top)))

pickle.dump(H, open(GRAPH_PKL, 'wb'))
print('saved rt_graph.pkl')
