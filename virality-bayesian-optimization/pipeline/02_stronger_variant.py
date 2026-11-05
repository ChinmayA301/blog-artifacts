"""
Stage 2 - stronger rule space + redundancy penalty, with a CELF baseline.

Design space (4 continuous knobs BO optimizes):
  a   - weight on z-scored out-degree        (how hub-greedy to be)
  b   - weight on z-scored k-core            (peripheral vs core)
  lam - redundancy penalty strength          (diversity pressure)
  hop - 1- or 2-hop neighbourhood for the redundancy discount
Seed builder: greedy forward selection over the top-3000 out-degree pool that
DISCOUNTS candidates inside the (undirected) neighbourhood of chosen seeds.
lam=0 reduces to a pure a/b ranking.

Baselines: shared top-degree, random seeds, and CELF (lazy greedy on 200 fixed
live-edge snapshots - see common.celf_snapshot - then scored with fresh,
independent IC simulations).
"""
import time
import numpy as np
import networkx as nx
import scipy.sparse as sp
from common import *

g = load_graph()
B = shared_baselines(g)
sim = IC(g, seed=11)

core = nx.core_number(g.H.to_undirected())
core_arr = np.array([core[n] for n in g.nodes], float)

POOL = 3000
pool = g.topk(g.outdeg, POOL)
pz = lambda x: (x - x.mean()) / x.std()
pdeg_z, pcore_z = pz(g.outdeg[pool]), pz(core_arr[pool])

A = sp.csr_matrix((np.ones(g.indices.size), g.indices, g.indptr), shape=(g.N, g.N))
U = ((A + A.T) > 0).tocsr()                 # undirected adjacency for the discount
uptr, uind = U.indptr, U.indices
def nbrs(ix):
    ix = np.atleast_1d(ix)
    return np.concatenate([uind[uptr[i]:uptr[i + 1]] for i in ix]) if len(ix) else np.array([], int)

def build_seeds(a, b, lam, hop):
    base = a * pdeg_z + b * pcore_z
    near = np.zeros(g.N, bool)
    avail = np.ones(POOL, bool)
    chosen = []
    for _ in range(K):
        score = base - lam * near[pool]
        score[~avail] = -1e9
        j = int(np.argmax(score))
        chosen.append(pool[j]); avail[j] = False
        nb = nbrs(pool[j]); near[nb] = True
        if hop >= 2:
            near[nbrs(nb)] = True
    return np.array(chosen)

def decode(theta):
    a, b, lam, hop = theta
    return build_seeds(a, b, lam, 1 if hop < 0.5 else 2)

def objective(theta):
    return sim.spread(decode(theta))

bounds = np.array([[0.2, 2.0], [0.0, 1.5], [0.0, 6.0], [0.0, 1.0]])
def sampler(r, n):
    return r.uniform(bounds[:, 0], bounds[:, 1], size=(n, 4))

print('running CELF on %d-node pool...' % POOL)
t0 = time.time()
celf_seeds, celf_saa = celf_snapshot(g, pool, n_worlds=200)
celf_final = sim.spread(celf_seeds, FINAL_SIMS)
celf_test = setlevel_test(sim, celf_seeds, B['td_reps'])
print('  CELF: snapshot estimate %.1f, fresh %d-sim %.1f (%.0fs); vs top-degree t=%.2f p=%.2g'
      % (celf_saa, FINAL_SIMS, celf_final, time.time() - t0, celf_test['t'], celf_test['p']))

N_EVALS, N_INIT, N_RUNS = budget(44), 6, 5
kern = default_kernel([0.5, 0.5, 1.0, 0.3])
print('BO vs random search: %d runs x %d evaluations each' % (N_RUNS, N_EVALS))
bo = run_many('BO', lambda s: run_bo(objective, sampler, N_INIT, N_EVALS, s, kernel=kern), N_RUNS, 0)
rs = run_many('RS', lambda s: run_random(objective, sampler, N_EVALS, s), N_RUNS, 200)

bo_f = reevaluate(bo, decode, sim); rs_f = reevaluate(rs, decode, sim)
bo_sum = summarize_runs('BO', bo_f, B['td_final'])
rs_sum = summarize_runs('RS', rs_f, B['td_final'])

best = bo[int(np.argmax(bo_f))]
shared = len(set(best['seeds']) & set(B['td_seeds']))
test = setlevel_test(sim, best['seeds'], B['td_reps'])
print('best BO seed set %.1f (theta=%s), shares %d/%d with top-degree; set-level t=%.2f p=%.2g'
      % (best['final'], np.round(best['best_x'], 2), shared, K, test['t'], test['p']))

save_stage('bo_strong_results.pkl', stage=2, label='Rule space + redundancy penalty (4-D)', dim=4,
           baselines=B, celf_final=celf_final, celf_saa=celf_saa, celf_seeds=celf_seeds, celf_test=celf_test,
           celf_shared=len(set(celf_seeds) & set(B['td_seeds'])),
           bo_curves=np.array([r['curve'] for r in bo]), rs_curves=np.array([r['curve'] for r in rs]),
           bo_finals=bo_f, rs_finals=rs_f, bo_summary=bo_sum, rs_summary=rs_sum,
           best_theta=best['best_x'], best_seeds=best['seeds'], best_final=best['final'],
           shared=shared, setlevel=test,
           ov_td=audience_overlap(g, B['td_seeds']), ov_bo=audience_overlap(g, best['seeds']),
           ov_celf=audience_overlap(g, celf_seeds),
           graph_nodes=g.N, graph_edges=g.n_edges)
