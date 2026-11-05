"""
common.py - shared configuration, graph loading, IC simulation, BO/random-search
loops and baselines for every stage of the experiment.

Why this module exists (audit fixes, Nov 2026):
  * ONE objective for every stage: same graph, same K, same IC_P, same MC.
  * ONE edge direction, asserted at load time (stale pickles fail loudly).
  * ONE top-degree baseline (global top-K by out-degree), evaluated once with a
    fixed RNG and shared by every stage and every chart.
  * BO and random-search curves indexed by the SAME evaluation count:
    curve[i] = best value seen after i+1 objective evaluations, for both.
  * Every run's best point is re-evaluated with fresh simulations, so we can
    report run-to-run spread instead of a single best-of-N number.
"""
import os, pickle, time, heapq, warnings
import numpy as np
import networkx as nx
from scipy.stats import norm, ttest_ind
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

# numpy 2.0 + macOS Accelerate emits spurious "encountered in matmul" warnings
# from GP prediction; GP hyperparameter fits hitting kernel bounds are routine in BO.
warnings.filterwarnings('ignore', message='.*encountered in matmul')
warnings.filterwarnings('ignore', module='sklearn.gaussian_process')

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_PKL = os.path.join(HERE, 'rt_graph.pkl')
BASELINE_PKL = os.path.join(HERE, 'baselines.pkl')

# Edge u -> v in rt_graph.pkl means "influence can flow from u to v", i.e. v
# retweeted u. SNAP's raw file lists (retweeter, retweeted), so
# 00_build_graph.py reverses it. Bump this tag if the convention ever changes.
DIRECTION = 'influence: retweeted -> retweeter (SNAP lines reversed)'

# ---- the single objective shared by all stages ----
K = 20                                        # seed budget
IC_P = float(os.environ.get('IC_P', 0.12))    # IC activation prob per edge
MC = int(os.environ.get('MC', 80))            # sims per BO/random-search evaluation
FINAL_SIMS = 400                              # fresh sims for re-evaluating winners
TD_SIMS = 4000                                # sims for the shared top-degree reference
SIG_REPS, SIG_SIMS = 30, 150                  # seed-set significance test
QUICK = os.environ.get('QUICK') == '1'        # smoke-test mode: tiny budgets


def budget(n):
    """Shrink evaluation budgets in QUICK mode so the pipeline can be smoke-tested."""
    return max(4, n // 8) if QUICK else n


class Graph:
    """Directed influence graph in integer-index CSR form."""
    def __init__(self, H):
        self.H = H
        self.nodes = np.array(H.nodes())
        self.N = len(self.nodes)
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        A = nx.to_scipy_sparse_array(H, nodelist=self.nodes, format='csr', weight=None)
        self.indptr = A.indptr.astype(np.int64)
        self.indices = A.indices.astype(np.int64)
        self.outdeg = np.diff(self.indptr).astype(float)   # = number of retweeters
        self.n_edges = H.number_of_edges()

    def topk(self, score, k=K, among=None):
        """Indices of the k highest-scoring nodes (optionally within `among`)."""
        if among is None:
            return np.argsort(-score, kind='stable')[:k]
        among = np.asarray(among)
        return among[np.argsort(-score[among], kind='stable')[:k]]


def load_graph():
    if not os.path.exists(GRAPH_PKL):
        raise SystemExit('rt_graph.pkl missing - run 00_build_graph.py first')
    H = pickle.load(open(GRAPH_PKL, 'rb'))
    if H.graph.get('direction') != DIRECTION:
        raise SystemExit('rt_graph.pkl was built with an old edge convention '
                         '(cascades ran retweeter -> retweeted). Rerun 00_build_graph.py.')
    return Graph(H)


class IC:
    """Independent Cascade simulator with its own RNG (vectorised per frontier)."""
    def __init__(self, g, p=IC_P, seed=0):
        self.g, self.p = g, p
        self.rng = np.random.default_rng(seed)
        self._active = np.zeros(g.N, dtype=bool)

    def spread(self, seeds, n_sims=MC):
        indptr, indices, p, rng, active = self.g.indptr, self.g.indices, self.p, self.rng, self._active
        seeds = np.unique(np.asarray(seeds, dtype=np.int64))
        total = 0
        for _ in range(n_sims):
            active[seeds] = True
            touched = [seeds]
            frontier = seeds
            while frontier.size:
                starts, ends = indptr[frontier], indptr[frontier + 1]
                lens = ends - starts
                m = int(lens.sum())
                if m == 0:
                    break
                offs = np.repeat(starts - (np.cumsum(lens) - lens), lens) + np.arange(m)
                cand = indices[offs[rng.random(m) < p]]
                cand = np.unique(cand)
                cand = cand[~active[cand]]
                active[cand] = True
                touched.append(cand)
                frontier = cand
            allt = np.concatenate(touched)
            total += allt.size
            active[allt] = False
        return total / n_sims

    def reps(self, seeds, reps=SIG_REPS, n_sims=SIG_SIMS):
        return np.array([self.spread(seeds, n_sims) for _ in range(reps)])


# ---------------------------------------------------------------- BO helpers
def expected_improvement(Xs, gp, y_best, xi=0.01):
    mu, sd = gp.predict(Xs, return_std=True)
    sd = np.maximum(sd, 1e-9)
    imp = mu - y_best - xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def default_kernel(length_scale=1.0):
    return lambda: (ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5)
                    + WhiteKernel(noise_level=5.0))


def run_bo(objective, sampler, n_init, n_evals, seed, kernel=None, n_cand=1000, restarts=2):
    """GP-EI BO. Returns curve where curve[i] = best y after i+1 evaluations
    (initial design included), so it lines up 1:1 with run_random()."""
    kernel = kernel or default_kernel()
    r = np.random.default_rng(seed)
    X = sampler(r, n_init)
    y = np.array([objective(x) for x in X])
    while len(y) < n_evals:
        gp = GaussianProcessRegressor(kernel=kernel(), normalize_y=True,
                                      n_restarts_optimizer=restarts, random_state=seed)
        gp.fit(X, y)
        cand = sampler(r, n_cand)
        x_next = cand[np.argmax(expected_improvement(cand, gp, y.max()))]
        X = np.vstack([X, x_next]); y = np.append(y, objective(x_next))
    return dict(curve=np.maximum.accumulate(y), X=X, y=y,
                best_x=X[np.argmax(y)], best_y=y.max())


def run_random(objective, sampler, n_evals, seed):
    """Random search with the same evaluation budget as run_bo()."""
    r = np.random.default_rng(seed)
    X = sampler(r, n_evals)
    y = np.array([objective(x) for x in X])
    return dict(curve=np.maximum.accumulate(y), X=X, y=y,
                best_x=X[np.argmax(y)], best_y=y.max())


def run_many(label, fn, n_runs, seed_offset):
    out = []
    for s in range(n_runs):
        t0 = time.time()
        res = fn(s + seed_offset)
        out.append(res)
        print('  %s run %d: best(noisy)=%.1f  [%.0fs]' % (label, s, res['best_y'], time.time() - t0))
    return out


def reevaluate(runs, decode, sim):
    """Fresh FINAL_SIMS evaluation of every run's best point (removes the
    winner's-curse bias of the noisy max, and exposes run-to-run spread)."""
    for res in runs:
        res['seeds'] = np.asarray(decode(res['best_x']))
        res['final'] = sim.spread(res['seeds'], FINAL_SIMS)
    finals = np.array([res['final'] for res in runs])
    return finals


def summarize_runs(name, finals, td_final):
    d = dict(mean=float(finals.mean()), std=float(finals.std(ddof=1)) if len(finals) > 1 else 0.0,
             min=float(finals.min()), max=float(finals.max()),
             n_runs=int(len(finals)), n_beat_td=int((finals > td_final).sum()))
    print('  %s (fresh %d-sim re-eval of each run\'s best): mean %.1f ± %.1f  [min %.1f, max %.1f]  '
          '%d/%d runs above top-degree (%.1f)' % (name, FINAL_SIMS, d['mean'], d['std'], d['min'], d['max'],
                                                 d['n_beat_td'], d['n_runs'], td_final))
    return d


def setlevel_test(sim, seeds, td_reps):
    """Welch t-test between Monte Carlo estimates of TWO FIXED seed sets.
    Supports 'this seed set beats top-degree', NOT 'the method beats top-degree'."""
    s = sim.reps(seeds)
    t, p = ttest_ind(s, td_reps, equal_var=False)
    return dict(mean=float(s.mean()), std=float(s.std(ddof=1)), t=float(t), p=float(p))


def audience_overlap(g, seeds):
    """1 - |union of seeds' direct audiences| / sum of audience sizes.
    Higher = seeds reach more of the same retweeters (more redundant)."""
    nb = [g.indices[g.indptr[s]:g.indptr[s + 1]] for s in seeds]
    tot = sum(len(a) for a in nb)
    return 1 - (len(np.unique(np.concatenate(nb))) / tot if tot else 0.0)


def save_stage(path, **kw):
    kw.update(K=K, IC_P=IC_P, MC=MC, FINAL_SIMS=FINAL_SIMS, QUICK=QUICK)
    pickle.dump(kw, open(os.path.join(HERE, path), 'wb'))
    print('saved', path)


# ---------------------------------------------------------------- baselines
def shared_baselines(g):
    """Top-degree / random-seed baselines, computed ONCE with a fixed RNG and
    cached, so every stage and chart compares against identical numbers."""
    key = (IC_P, MC, TD_SIMS, g.N, g.n_edges, DIRECTION)
    if os.path.exists(BASELINE_PKL):
        b = pickle.load(open(BASELINE_PKL, 'rb'))
        if b.get('key') == key:
            return b
    print('computing shared baselines (cached to baselines.pkl)...')
    sim = IC(g, seed=12345)
    td = g.topk(g.outdeg)
    # High-precision reference (10x FINAL_SIMS) so that "runs above top-degree"
    # counts are not driven by one lucky/unlucky 400-sim draw of the baseline.
    td_final = sim.spread(td, TD_SIMS)
    td_reps = sim.reps(td)
    r = np.random.default_rng(999)
    rand = np.array([sim.spread(r.choice(g.N, K, replace=False), 120) for _ in range(25)])
    b = dict(key=key, td_seeds=td, td_final=td_final, td_reps=td_reps,
             td_mean=float(td_reps.mean()), td_std=float(td_reps.std(ddof=1)),
             rand_mean=float(rand.mean()), rand_std=float(rand.std(ddof=1)),
             K=K, IC_P=IC_P, MC=MC, N=g.N, E=g.n_edges)
    pickle.dump(b, open(BASELINE_PKL, 'wb'))
    print('  top-degree=%.1f (reps %.1f ± %.1f)  random seeds=%.1f ± %.1f'
          % (td_final, b['td_mean'], b['td_std'], b['rand_mean'], b['rand_std']))
    return b


def celf_snapshot(g, pool, k=K, p=IC_P, n_worlds=200, seed=777):
    """CELF (lazy greedy) on a FIXED set of sampled live-edge graphs.

    The old CELF re-simulated every marginal gain with 20-40 fresh cascades,
    which is too noisy for lazy evaluation (stale upper bounds become wrong).
    Here spread is the sample-average over n_worlds live-edge snapshots, which
    is exactly monotone submodular, so lazy greedy is valid. The selected set
    is then scored with independent IC simulations by the caller.
    """
    r = np.random.default_rng(seed)
    N, indptr, indices = g.N, g.indptr, g.indices
    rows = np.repeat(np.arange(N), np.diff(indptr))
    worlds = []
    for _ in range(n_worlds):
        live = r.random(indices.size) < p
        w_ptr = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows[live], minlength=N), out=w_ptr[1:])
        worlds.append((w_ptr, indices[live]))
    covered = np.zeros((n_worlds, N), dtype=bool)

    def gain(u, commit=False):
        tot = 0
        for w, (wp, wi) in enumerate(worlds):
            cov = covered[w]
            if cov[u]:
                continue
            seen = {u}; stack = [u]
            while stack:
                x = stack.pop()
                for y in wi[wp[x]:wp[x + 1]]:
                    if not cov[y] and y not in seen:
                        seen.add(y); stack.append(y)
            tot += len(seen)
            if commit:
                cov[list(seen)] = True
        return tot / n_worlds

    t0 = time.time()
    heap = [(-gain(int(u)), int(u), 0) for u in pool]
    heapq.heapify(heap)
    print('  CELF singleton gains for %d nodes x %d worlds in %.0fs' % (len(pool), n_worlds, time.time() - t0))
    S, saa = [], 0.0
    while len(S) < k:
        neg, u, it = heapq.heappop(heap)
        if it == len(S):
            saa += gain(u, commit=True)
            S.append(u)
        else:
            heapq.heappush(heap, (-gain(u), u, len(S)))
    return np.array(S), saa
