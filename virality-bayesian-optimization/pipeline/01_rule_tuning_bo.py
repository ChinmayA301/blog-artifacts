"""
Engineering Virality with Bayesian Optimization
Real pipeline on the SNAP Higgs retweet network (largest WCC: 223,833 nodes).

Experiment: select a seed set of k=20 nodes to maximize expected influence
spread under the Independent Cascade (IC) model. We compare:
  - Random search over seed sets
  - Bayesian Optimization over a low-dim *strategy* parameterization
  - Fixed heuristic baselines: top-degree, top-PageRank, top-k-core, random
BO doesn't search raw node IDs (combinatorial); it searches a continuous
mixing weight over centrality rankings, which is a realistic, smooth design space.
"""
import gzip, pickle, time
import numpy as np
import networkx as nx
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

rng = np.random.default_rng(7)
H = pickle.load(open('rt_graph.pkl','rb'))
nodes = np.array(H.nodes())
N = len(nodes)
K = 20          # seed budget
IC_P = 0.05     # IC activation probability per edge
MC = 60         # Monte Carlo cascades per evaluation (noisy objective -> good for BO)

# ---- Precompute centrality rankings (the design basis) ----
t0 = time.time()
deg = dict(H.out_degree())
deg_arr = np.array([deg[n] for n in nodes], dtype=float)
pr = nx.pagerank(H, alpha=0.85, max_iter=50, tol=1e-4)
pr_arr = np.array([pr[n] for n in nodes], dtype=float)
core = nx.core_number(H.to_undirected())
core_arr = np.array([core[n] for n in nodes], dtype=float)
print('centralities computed in %.1fs' % (time.time()-t0))

def zscore(x):
    s = x.std()
    return (x - x.mean()) / (s if s>0 else 1.0)
Zd, Zp, Zc = zscore(deg_arr), zscore(pr_arr), zscore(core_arr)

# adjacency for fast IC
succ = {n:list(H.successors(n)) for n in nodes}
node_index = {n:i for i,n in enumerate(nodes)}

def independent_cascade(seed_nodes, n_sims=MC):
    """Return mean activated count over n_sims IC runs."""
    totals = 0
    for _ in range(n_sims):
        active = set(seed_nodes)
        frontier = list(seed_nodes)
        while frontier:
            nxt = []
            for u in frontier:
                for v in succ[u]:
                    if v not in active and rng.random() < IC_P:
                        active.add(v); nxt.append(v)
            frontier = nxt
        totals += len(active)
    return totals / n_sims

def seeds_from_weights(w):
    """Map a 3-vector of weights -> top-K nodes by blended centrality score."""
    w = np.clip(w, 0, 1)
    score = w[0]*Zd + w[1]*Zp + w[2]*Zc
    top = np.argpartition(-score, K)[:K]
    return [nodes[i] for i in top]

def objective(w):
    return independent_cascade(seeds_from_weights(w))

# ---- Fixed heuristic baselines ----
def topk_by(arr):
    idx = np.argpartition(-arr, K)[:K]
    return [nodes[i] for i in idx]

print('evaluating heuristic baselines...')
base_results = {}
for name, seeds in [
    ('Top-degree',  topk_by(deg_arr)),
    ('Top-PageRank',topk_by(pr_arr)),
    ('Top-k-core',  topk_by(core_arr)),
]:
    vals = [independent_cascade(seeds, n_sims=200) for _ in range(1)]
    base_results[name] = independent_cascade(seeds, n_sims=400)
# random seed baseline: average of many random sets
rand_vals = [independent_cascade(list(rng.choice(nodes, K, replace=False)), n_sims=120) for _ in range(25)]
base_results['Random seeds'] = np.mean(rand_vals)
base_results['_random_std'] = np.std(rand_vals)
print('baselines:', {k:round(v,1) for k,v in base_results.items()})

# ---- Random search vs BO, multiple seeds for confidence bands ----
N_ITERS = 40
N_RUNS  = 5
bounds = np.array([[0,1],[0,1],[0,1]])

def expected_improvement(Xs, gp, y_best, xi=0.01):
    mu, sd = gp.predict(Xs, return_std=True)
    sd = np.maximum(sd, 1e-9)
    imp = mu - y_best - xi
    z = imp/sd
    return imp*norm.cdf(z) + sd*norm.pdf(z)

def run_bo(seed):
    r = np.random.default_rng(seed)
    X = r.uniform(0,1,size=(5,3))                 # initial design
    y = np.array([objective(x) for x in X])
    best_curve = [y.max()]
    for it in range(N_ITERS-5):
        kernel = ConstantKernel(1.0)*Matern(length_scale=0.5, nu=2.5) + WhiteKernel(noise_level=5.0)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed)
        gp.fit(X, y)
        cand = r.uniform(0,1,size=(800,3))
        ei = expected_improvement(cand, gp, y.max())
        x_next = cand[np.argmax(ei)]
        y_next = objective(x_next)
        X = np.vstack([X, x_next]); y = np.append(y, y_next)
        best_curve.append(y.max())
    return np.array(best_curve), X[np.argmax(y)], y.max()

def run_random(seed):
    r = np.random.default_rng(seed+100)
    best_curve=[]; best=-1
    for it in range(N_ITERS-4):
        v = objective(r.uniform(0,1,size=3))
        best = max(best, v); best_curve.append(best)
    return np.array(best_curve)

print('running BO (%d runs x %d iters)...' % (N_RUNS, N_ITERS))
bo_curves=[]; bo_best_w=None; bo_best_val=-1
for s in range(N_RUNS):
    c, w, v = run_bo(s)
    bo_curves.append(c)
    if v>bo_best_val: bo_best_val, bo_best_w = v, w
    print('  BO run %d: best=%.1f w=%s' % (s, v, np.round(w,2)))
rand_curves=[run_random(s) for s in range(N_RUNS)]

# align curve lengths
L = min(min(len(c) for c in bo_curves), min(len(c) for c in rand_curves))
bo_curves=np.array([c[:L] for c in bo_curves])
rand_curves=np.array([c[:L] for c in rand_curves])

# BO's discovered seed set, evaluated at high precision
bo_seeds = seeds_from_weights(bo_best_w)
bo_final = independent_cascade(bo_seeds, n_sims=400)
print('BO best weights:', np.round(bo_best_w,3), '-> spread=%.1f' % bo_final)

pickle.dump({
    'bo_curves':bo_curves,'rand_curves':rand_curves,
    'base_results':base_results,'bo_final':bo_final,
    'bo_best_w':bo_best_w,'K':K,'IC_P':IC_P,'N':N,
    'graph_nodes':H.number_of_nodes(),'graph_edges':H.number_of_edges(),
}, open('bo_results.pkl','wb'))
print('saved.')
