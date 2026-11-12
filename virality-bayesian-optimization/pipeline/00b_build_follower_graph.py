"""
00b_build_follower_graph.py
Build the FOLLOWER (exposure) graph for the Higgs data.

Input : higgs-social_network.edgelist.gz   (SNAP Higgs; line "A B" = A follows B)
Output: follower_graph.npz  (largest weakly-connected component, CSR arrays)

Why a second graph: the retweet graph's edges are retweets that already
happened, so simulating cascades over it spreads simulated diffusion along
realised outcomes. Exposure on Twitter runs along follow edges, so the
diffusion edge is followee -> follower. (Orientation checked against the
activity log: 61% of retweets have the retweeter following the source,
13% the reverse.)

The graph is stored as numpy CSR rather than a networkx DiGraph: 14.9M edges
would need many GB as Python objects.
"""
import os, sys, gzip
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, 'higgs-social_network.edgelist.gz')

E = np.loadtxt(gzip.open(src, 'rt'), dtype=np.int64)       # (follower, followee)
follower, followee = E[:, 0], E[:, 1]
ids = np.unique(np.concatenate([follower, followee]))
pos = np.full(ids.max() + 1, -1, dtype=np.int64); pos[ids] = np.arange(len(ids))
u, v = pos[followee], pos[follower]                          # influence: followee -> follower
n = len(ids)
A = sp.csr_matrix((np.ones(len(u), dtype=np.int8), (u, v)), shape=(n, n))
A.sum_duplicates()
print('follower graph: nodes=%d edges=%d' % (n, A.nnz))

_, lab = connected_components(A, directed=True, connection='weak')
big = np.bincount(lab).argmax()
keep = np.flatnonzero(lab == big)
A = A[keep][:, keep].tocsr()
ids = ids[keep]
print('largest WCC: nodes=%d edges=%d' % (A.shape[0], A.nnz))

followers = np.diff(A.indptr)
print('max followers: %d (node %d)' % (followers.max(), ids[followers.argmax()]))
np.savez_compressed(os.path.join(here, 'follower_graph.npz'), indptr=A.indptr.astype(np.int64),
                    indices=A.indices.astype(np.int64), ids=ids,
                    direction='influence: followee -> follower (SNAP lines reversed)')
print('saved follower_graph.npz')
