"""
Stage 3 - free node-level BO, naive (high-dimensional) parameterisation.

  1. Candidate pool: top 1500 nodes by out-degree; keep the largest connected
     component of their induced (undirected) subgraph.
  2. Spectral-embed that component into D dims (Laplacian eigenmaps).
  3. BO searches K anchor points in [0,1]^D  ->  K*D continuous dims. Each
     anchor decodes to its nearest unused embedded node -> a seed set.
Baseline: the SAME shared global top-degree set used by every stage.
"""
import time
import numpy as np
import networkx as nx
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import NearestNeighbors
from common import *

g = load_graph()
B = shared_baselines(g)
sim = IC(g, seed=23)
pad_rng = np.random.default_rng(230)

pool = g.topk(g.outdeg, 1500)
sub = g.H.subgraph(g.nodes[pool]).to_undirected()
cc = max(nx.connected_components(sub), key=len)
sub_cc = sub.subgraph(cc)
emb_ids = np.array(sub_cc.nodes())
emb_ix = np.array([g.idx[n] for n in emb_ids])
A = nx.to_scipy_sparse_array(sub_cc, nodelist=emb_ids, format='csr', dtype=float)
A.indices = A.indices.astype(np.int32); A.indptr = A.indptr.astype(np.int32)
D = 6
t0 = time.time()
EMB = spectral_embedding(A, n_components=D, random_state=0)
EMB = (EMB - EMB.min(0)) / (np.ptp(EMB, axis=0) + 1e-9)
nn = NearestNeighbors(n_neighbors=1).fit(EMB)
td_in_pool = len(set(B['td_seeds']) & set(emb_ix))
print('embedding: %d of 1500 pool nodes in %dD (%.1fs); %d/%d top-degree seeds are in it'
      % (len(emb_ix), D, time.time() - t0, td_in_pool, K))

def decode(flat):
    _, idxs = nn.kneighbors(flat.reshape(K, D), n_neighbors=8)
    seeds, used = [], set()
    for row in idxs:
        for j in row:
            if j not in used:
                used.add(j); seeds.append(emb_ix[j]); break
    while len(seeds) < K:                       # pad if collisions reduced count
        j = int(pad_rng.integers(len(emb_ix)))
        if j not in used:
            used.add(j); seeds.append(emb_ix[j])
    return np.array(seeds[:K])

def objective(flat):
    return sim.spread(decode(flat))

DIM = K * D
def sampler(r, n):
    return r.uniform(0, 1, size=(n, DIM))

N_EVALS, N_INIT, N_RUNS = budget(60), 12, 4
print('BO vs random search: %d runs x %d evaluations each, %d-D' % (N_RUNS, N_EVALS, DIM))
bo = run_many('BO', lambda s: run_bo(objective, sampler, N_INIT, N_EVALS, s, n_cand=1500, restarts=1),
              N_RUNS, 0)
rs = run_many('RS', lambda s: run_random(objective, sampler, N_EVALS, s), N_RUNS, 300)

bo_f = reevaluate(bo, decode, sim); rs_f = reevaluate(rs, decode, sim)
bo_sum = summarize_runs('BO', bo_f, B['td_final'])
rs_sum = summarize_runs('RS', rs_f, B['td_final'])

best = bo[int(np.argmax(bo_f))]
shared = len(set(best['seeds']) & set(B['td_seeds']))
test = setlevel_test(sim, best['seeds'], B['td_reps'])
print('best BO seed set %.1f, shares %d/%d with top-degree; set-level t=%.2f p=%.2g'
      % (best['final'], shared, K, test['t'], test['p']))

save_stage('bo_freenode_results.pkl', stage=3, label='Free-node, naive (%d-D embedding)' % DIM, dim=DIM, D=D,
           pool=len(emb_ix), td_in_pool=td_in_pool, baselines=B,
           bo_curves=np.array([r['curve'] for r in bo]), rs_curves=np.array([r['curve'] for r in rs]),
           bo_finals=bo_f, rs_finals=rs_f, bo_summary=bo_sum, rs_summary=rs_sum,
           best_seeds=best['seeds'], best_final=best['final'], shared=shared, setlevel=test,
           ov_td=audience_overlap(g, B['td_seeds']), ov_bo=audience_overlap(g, best['seeds']),
           graph_nodes=g.N, graph_edges=g.n_edges)
