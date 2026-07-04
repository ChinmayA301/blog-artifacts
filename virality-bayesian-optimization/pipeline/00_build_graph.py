"""
00_build_graph.py
Build the retweet diffusion graph used by all BO experiments.

Input : higgs-retweet_network.edgelist.gz  (SNAP Higgs dataset)
Output: rt_graph.pkl  (largest weakly-connected component, ~223,833 nodes)

Download the source file from:
  https://snap.stanford.edu/data/higgs-twitter.html
  (file: higgs-retweet_network.edgelist.gz)
Place it next to this script, or pass its path as argv[1].
"""
import sys, gzip, pickle
import networkx as nx

src = sys.argv[1] if len(sys.argv) > 1 else 'higgs-retweet_network.edgelist.gz'

G = nx.DiGraph()
with gzip.open(src, 'rt') as f:
    for line in f:
        u, v, w = line.split()
        # directed: u retweeted v -> influence flows v -> u
        G.add_edge(int(u), int(v), weight=int(w))

print('retweet graph: nodes=%d edges=%d' % (G.number_of_nodes(), G.number_of_edges()))

# largest weakly-connected component = clean diffusion substrate
wcc = max(nx.weakly_connected_components(G), key=len)
H = G.subgraph(wcc).copy()
print('largest WCC: nodes=%d edges=%d' % (H.number_of_nodes(), H.number_of_edges()))

pickle.dump(H, open('rt_graph.pkl', 'wb'))
print('saved rt_graph.pkl')
