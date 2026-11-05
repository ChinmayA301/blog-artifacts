"""
Stage 4 - constrained free-node BO: diverse SUBSETS of strong nodes.

Pool: top 400 nodes by out-degree (the shared top-degree set is its top 20).
Decoder: greedy pick by  z(out-degree) + bias - lam * [in a chosen seed's
neighbourhood],  where bias = z @ R is a per-node logit bias from a fixed
random projection R (P latent dims) and lam = 4 * theta[P].
Search space: P latent dims + 1 diversity weight = P+1 dims (13 with P=12).

Controls added Nov 2026 so a win can be attributed:
  * random search in the SAME 13-D space with the SAME evaluation budget;
  * decoder-only: z = 0 (no learned bias), lam swept on a grid with the same
    MC objective, best grid point re-evaluated fresh. If BO does not beat this,
    the gain belongs to the degree+diversity decoder, not to BO.
  * every run's best is re-evaluated fresh -> run-to-run spread is reported.
"""
import numpy as np
from common import *

g = load_graph()
B = shared_baselines(g)
sim = IC(g, seed=31)
proj_rng = np.random.default_rng(31)

POOL = 400
pool = g.topk(g.outdeg, POOL)
assert set(B['td_seeds']) <= set(pool)
pdeg_z = (g.outdeg[pool] - g.outdeg[pool].mean()) / g.outdeg[pool].std()
sub = g.H.subgraph(g.nodes[pool]).to_undirected()
pos = {g.idx[n]: j for j, n in enumerate(g.nodes[pool])}
neigh = [np.array([pos[g.idx[m]] for m in sub.neighbors(n)], int) for n in g.nodes[pool]]

P = 12
R = proj_rng.normal(size=(P, POOL)) / np.sqrt(P)
DIM = P + 1

def decode(theta):
    z, lam = theta[:P], theta[P] * 4
    score0 = pdeg_z + z @ R
    near = np.zeros(POOL, bool); avail = np.ones(POOL, bool); chosen = []
    for _ in range(K):
        sc = score0 - lam * near
        sc[~avail] = -1e9
        j = int(np.argmax(sc)); chosen.append(pool[j]); avail[j] = False
        near[neigh[j]] = True
    return np.array(chosen)

def objective(theta):
    return sim.spread(decode(theta))

def sampler(r, n):
    return np.hstack([r.normal(size=(n, P)), r.uniform(0, 1, size=(n, 1))])

# ---- decoder-only control: no learned bias, sweep the diversity weight ----
grid = np.linspace(0, 1, 11)
grid_vals = np.array([objective(np.r_[np.zeros(P), u]) for u in grid])
dec_theta = np.r_[np.zeros(P), grid[int(np.argmax(grid_vals))]]
dec_seeds = decode(dec_theta)
dec_final = sim.spread(dec_seeds, FINAL_SIMS)
dec_test = setlevel_test(sim, dec_seeds, B['td_reps'])
print('decoder-only (z=0): best lam=%.1f -> fresh %.1f vs top-degree %.1f (set-level p=%.2g)'
      % (dec_theta[P] * 4, dec_final, B['td_final'], dec_test['p']))

N_EVALS, N_INIT, N_RUNS = budget(84), 14, 5
print('BO vs random search: %d runs x %d evaluations each, %d-D' % (N_RUNS, N_EVALS, DIM))
bo = run_many('BO', lambda s: run_bo(objective, sampler, N_INIT, N_EVALS, s, n_cand=1500, restarts=1),
              N_RUNS, 0)
rs = run_many('RS', lambda s: run_random(objective, sampler, N_EVALS, s), N_RUNS, 400)

bo_f = reevaluate(bo, decode, sim); rs_f = reevaluate(rs, decode, sim)
bo_sum = summarize_runs('BO', bo_f, B['td_final'])
rs_sum = summarize_runs('RS', rs_f, B['td_final'])
print('  BO runs above decoder-only (%.1f): %d/%d' % (dec_final, int((bo_f > dec_final).sum()), N_RUNS))

best = bo[int(np.argmax(bo_f))]
shared = len(set(best['seeds']) & set(B['td_seeds']))
test = setlevel_test(sim, best['seeds'], B['td_reps'])
print('best BO seed set %.1f, shares %d/%d with top-degree; set-level t=%.2f p=%.2g'
      % (best['final'], shared, K, test['t'], test['p']))

save_stage('bo_constrained_results.pkl', stage=4, label='Free-node, constrained (%d-D)' % DIM, dim=DIM, P=P,
           pool=POOL, baselines=B,
           dec_grid=grid * 4, dec_grid_vals=grid_vals, dec_final=dec_final, dec_seeds=dec_seeds,
           dec_test=dec_test, dec_shared=len(set(dec_seeds) & set(B['td_seeds'])),
           bo_curves=np.array([r['curve'] for r in bo]), rs_curves=np.array([r['curve'] for r in rs]),
           bo_finals=bo_f, rs_finals=rs_f, bo_summary=bo_sum, rs_summary=rs_sum,
           best_theta=best['best_x'], best_seeds=best['seeds'], best_final=best['final'],
           shared=shared, setlevel=test,
           ov_td=audience_overlap(g, B['td_seeds']), ov_bo=audience_overlap(g, best['seeds']),
           graph_nodes=g.N, graph_edges=g.n_edges)
