"""
Stage 1 - rule tuning.
BO searches a 3-D mixing weight over z-scored centrality rankings
(out-degree, PageRank, k-core); the seed set is the top-K by blended score.
Compared against random search with the SAME number of objective evaluations,
and against fixed heuristics.

Objective (shared by all stages, see common.py): mean IC spread, k=K,
p=IC_P, MC sims per evaluation.
"""
import numpy as np
import networkx as nx
from common import *

g = load_graph()
B = shared_baselines(g)
sim = IC(g, seed=7)

# Centrality basis. Out-degree = number of retweeters. PageRank is computed on
# the ENDORSEMENT graph (retweeter -> retweeted, i.e. H reversed) so it ranks
# accounts that are retweeted by well-retweeted accounts - the influence
# reading of PageRank. On the diffusion graph itself it would rank sinks.
pr = nx.pagerank(g.H.reverse(copy=False), alpha=0.85, max_iter=100, tol=1e-6)
pr_arr = np.array([pr[n] for n in g.nodes])
core = nx.core_number(g.H.to_undirected())
core_arr = np.array([core[n] for n in g.nodes], float)

def z(x):
    s = x.std(); return (x - x.mean()) / (s if s > 0 else 1.0)
Zd, Zp, Zc = z(g.outdeg), z(pr_arr), z(core_arr)

def decode(w):
    w = np.clip(w, 0, 1)
    return g.topk(w[0] * Zd + w[1] * Zp + w[2] * Zc)

def objective(w):
    return sim.spread(decode(w))

def sampler(r, n):
    return r.uniform(0, 1, size=(n, 3))

heur = {'Top-degree': B['td_final'],
        'Top-PageRank': sim.spread(g.topk(pr_arr), FINAL_SIMS),
        'Top-k-core': sim.spread(g.topk(core_arr), FINAL_SIMS),
        'Random seeds': B['rand_mean']}
print('heuristics:', {k: round(v, 1) for k, v in heur.items()})

N_EVALS, N_INIT, N_RUNS = budget(40), 5, 5
print('BO vs random search: %d runs x %d evaluations each' % (N_RUNS, N_EVALS))
bo = run_many('BO', lambda s: run_bo(objective, sampler, N_INIT, N_EVALS, s,
                                     kernel=default_kernel(0.5), n_cand=800), N_RUNS, 0)
rs = run_many('RS', lambda s: run_random(objective, sampler, N_EVALS, s), N_RUNS, 100)

bo_f = reevaluate(bo, decode, sim); rs_f = reevaluate(rs, decode, sim)
bo_sum = summarize_runs('BO', bo_f, B['td_final'])
rs_sum = summarize_runs('RS', rs_f, B['td_final'])

best = bo[int(np.argmax(bo_f))]
shared = len(set(best['seeds']) & set(B['td_seeds']))
test = setlevel_test(sim, best['seeds'], B['td_reps'])
print('best BO seed set: %.1f, shares %d/%d seeds with top-degree; set-level Welch t=%.2f p=%.2g'
      % (best['final'], shared, K, test['t'], test['p']))

save_stage('bo_results.pkl', stage=1, label='Rule-tuning (3-D centrality blend)', dim=3,
           heur=heur, baselines=B,
           bo_curves=np.array([r['curve'] for r in bo]), rs_curves=np.array([r['curve'] for r in rs]),
           bo_finals=bo_f, rs_finals=rs_f, bo_summary=bo_sum, rs_summary=rs_sum,
           best_w=best['best_x'], best_seeds=best['seeds'], best_final=best['final'],
           shared=shared, setlevel=test,
           ov_td=audience_overlap(g, B['td_seeds']), ov_bo=audience_overlap(g, best['seeds']),
           graph_nodes=g.N, graph_edges=g.n_edges)
