"""
Stage 6 - when do adaptive, sample-efficient methods earn their place?

The first five stages sit in one corner of the problem: the spread model is
KNOWN and evaluating a seed set is CHEAP (a simulation). Greedy has a
near-optimality guarantee there, and nothing beat top-degree by more than ~1%.

This stage moves to the opposite corner: the per-account transmissibility is
UNKNOWN, and every evaluation is ONE real campaign (one cascade), run over
T rounds. It is the online / combinatorial-bandit influence-maximization setting
(Chen, Wang & Yuan 2013; Wen et al. 2017), here with source-level parameter
sharing so it scales to the 14.9M-edge Higgs follower graph.

Ground-truth worlds (follower graph, edge u->v = v follows u):
  p_uv = min(1, c_u / max(followees(v), F)),   F = 20 (validation/calibration.json)
  W1 homogeneous   c_u = c                     (c = 0.726, whole-week calibration)
  W3 mild           as W2 with 10x stronger shrinkage toward c
  W2 heterogeneous c_u estimated per account from the real activity log:
                   c_u = (A_u + a) / (M_u + a / c)
                   for accounts that posted about Higgs that week
                   (A_u = distinct follower-retweeters, M_u = weighted-cascade exposure
                   mass, a = gamma-Poisson dispersion fitted by marginal likelihood);
                   c_u = c for accounts that did not post (no evidence either way).
  W2 is a plausible, data-shaped heterogeneity scenario, not measured truth: an
  account's realised retweet yield mixes transmissibility with luck and with
  whether it posted at all.

Policies, each choosing K=20 seeds per round and observing that round's cascade:
  oracle       greedy with the TRUE c_u (known dynamics; the reference)
  top-degree   top-K by followers every round (needs no model)
  prior-greedy greedy assuming every c_u = c (aggregate calibration only)
  CUCB         greedy on an upper confidence bound of each c_u
  Thompson     greedy on a posterior sample of each c_u
  BO           GP-EI over the stage-1 3-D centrality blend, one cascade per evaluation
Learners observe edge-level outcomes (who was exposed, who retweeted whom), and
update a Gamma posterior per account: S_u ~ Poisson(c_u * M_u), prior
Gamma(a, a / c), i.e. they know the aggregate calibration and the dispersion,
not which accounts are strong.

Greedy oracle (fast): marginal gain of u = sum over followers v of
P(v not yet reached) * p_uv * (1 + h_v), with h_v = sum_w p_vw the expected
direct reach of v. A two-hop approximation, fine for shallow cascades (observed
depth 0.056); checked against snapshot CELF below.

Outputs: unknown_dynamics.pkl (pipeline/), printed summary.
"""
import os, sys, gzip, time, pickle, json
import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln
from scipy.stats import gamma as gamma_dist
os.environ.setdefault('GRAPH', 'follower')
from common import (HERE, CALIBRATION_JSON, FOLLOWER_NPZ, K, Graph, IC, pagerank_endorsement, coreness,
                    default_kernel, expected_improvement, celf_snapshot, QUICK)
from sklearn.gaussian_process import GaussianProcessRegressor

T = 12 if QUICK else int(os.environ.get('T', 50))
N_RUNS = 2 if QUICK else int(os.environ.get('N_RUNS', 5))
POOL = 3000
EVAL_SIMS = 100 if QUICK else 300
t_start = time.time()

cal = json.load(open(CALIBRATION_JSON))
F = cal['chosen']['floor']; C = cal['full']['wc_c_by_floor'][str(F)]
z = np.load(FOLLOWER_NPZ)
g = Graph.from_csr(z['indptr'], z['indices'], z['ids'])
n, m = g.N, g.n_edges
src = np.repeat(np.arange(n), np.diff(g.indptr))
followees = np.bincount(g.indices, minlength=n)
w_e = 1.0 / np.maximum(followees, F)[g.indices]          # per-edge attention weight
mass = np.bincount(src, weights=w_e, minlength=n)        # M_u if every follower is exposed
pool = g.topk(g.outdeg, POOL)
print('graph %d nodes / %d edges; F=%d c=%.3f; T=%d rounds x %d runs' % (n, m, F, C, T, N_RUNS))

# ---------------------------------------------------- W2: per-account c_u from data
pos = np.full(int(g.nodes.max()) + 1, -1, np.int64); pos[g.nodes] = np.arange(n)
raw = np.loadtxt(gzip.open(os.path.join(HERE, 'higgs-activity_time.txt.gz'), 'rt'),
                 dtype=str, usecols=(0, 1, 3))
a_, b_ = raw[:, 0].astype(np.int64), raw[:, 1].astype(np.int64)
ok = (a_ <= g.nodes.max()) & (b_ <= g.nodes.max())
a_, b_, kind = pos[a_[ok]], pos[b_[ok]], raw[ok, 2]
ok = (a_ >= 0) & (b_ >= 0); a_, b_, kind = a_[ok], b_[ok], kind[ok]
rt = kind == 'RT'
# 'posted' = authored any Higgs action, or was retweeted/replied to (so had a Higgs post).
# Accounts merely mentioned, or absent, carry no evidence and keep the prior c.
active = np.zeros(n, bool); active[a_] = True; active[b_[rt | (kind == 'RE')]] = True
pairs = np.unique(np.stack([b_[rt], a_[rt]], 1), axis=0)           # (source, retweeter)
keys = np.sort(src * n + g.indices)
kq = pairs[:, 0] * n + pairs[:, 1]; i = np.searchsorted(keys, kq); i[i >= m] = 0
fpairs = pairs[keys[i] == kq]                                      # retweeter follows source
A = np.bincount(fpairs[:, 0], minlength=n).astype(float)
act = np.where(active & (mass > 0))[0]
print('posting sources %d; pooled c over them %.3f (calibration %.3f)' % (len(act), A[act].sum() / mass[act].sum(), C))

def nb_loglik(alpha):            # gamma-Poisson marginal likelihood, prior mean C
    y, mu = A[act], C * mass[act]
    return (gammaln(y + alpha) - gammaln(alpha) + alpha * np.log(alpha / (alpha + mu))
            + y * np.log(mu / (alpha + mu))).sum()
alphas = np.exp(np.linspace(np.log(0.01), np.log(100), 81))
ALPHA = float(alphas[np.argmax([nb_loglik(al) for al in alphas])])
c_w2 = np.full(n, C); c_w2[act] = (A[act] + ALPHA) / (mass[act] + ALPHA / C)
print('W2 dispersion a=%.3f; c_u among pool: median %.3f, p10 %.3f, p90 %.3f'
      % (ALPHA, *np.percentile(c_w2[pool], [50, 10, 90])))
# W3: same per-account evidence, 10x stronger shrinkage toward c (milder heterogeneity)
c_w3 = np.full(n, C); c_w3[act] = (A[act] + 10 * ALPHA) / (mass[act] + 10 * ALPHA / C)
print('W3 (a x10): c_u among pool: median %.3f, p10 %.3f, p90 %.3f' % tuple(np.percentile(c_w3[pool], [50, 10, 90])))
WORLDS = {'W1_homogeneous': np.full(n, C), 'W3_mild_heterogeneity': c_w3, 'W2_heterogeneous': c_w2}
# Dose-response (GAMMAS="0.25,0.5"): c_u = s * C * (c_u^W2 / C)^gamma, s keeps the
# exposure-weighted mean over posting accounts equal to W2's. gamma=0 is W1, gamma=1 is W2.
GAMMAS = [float(x) for x in os.environ.get('GAMMAS', '').split(',') if x]
if GAMMAS:
    WORLDS = {}
    for gm in GAMMAS:
        cg = C * (c_w2 / C) ** gm
        cg[act] *= (mass[act] * c_w2[act]).sum() / (mass[act] * cg[act]).sum()
        WORLDS['Wgamma_%g' % gm] = cg
POLICIES = os.environ.get('POLICIES', 'oracle,top-degree,prior-greedy,CUCB,Thompson,BO').split(',')

# ---------------------------------------------------- simulation with feedback
def eprob(c):
    return np.minimum(1.0, c[src] * w_e)

def cascade(ep, seeds, rng):
    """One IC cascade. Returns (size, S, M): per-source coin successes and exposure mass
    over the edges each activated source tried (targets inactive at that step)."""
    indptr, indices = g.indptr, g.indices
    on = np.zeros(n, bool); seeds = np.unique(seeds); on[seeds] = True
    S = np.zeros(n); M = np.zeros(n); frontier = seeds; size = seeds.size
    while frontier.size:
        st, en = indptr[frontier], indptr[frontier + 1]; ln = en - st; tot = int(ln.sum())
        if tot == 0: break
        offs = np.repeat(st - (np.cumsum(ln) - ln), ln) + np.arange(tot)
        tg = indices[offs]; fresh = ~on[tg]; offs, tg = offs[fresh], tg[fresh]
        par = src[offs]
        hit = rng.random(offs.size) < ep[offs]
        np.add.at(M, par, w_e[offs]); np.add.at(S, par[hit], 1)
        new = np.unique(tg[hit]); on[new] = True; size += new.size; frontier = new
    return size, S, M

# ---------------------------------------------------- fast greedy oracle
pool_off = np.concatenate([np.arange(g.indptr[u], g.indptr[u + 1]) for u in pool])
pool_ptr = np.concatenate([[0], np.cumsum(g.outdeg[pool].astype(np.int64))])
pool_tgt = g.indices[pool_off]; pool_w = w_e[pool_off]
print('pool of %d holds %d edges' % (POOL, pool_off.size))

def greedy(c):
    ep = eprob(c)
    h = np.bincount(src, weights=ep, minlength=n)                  # expected direct reach
    pp = np.minimum(1.0, np.repeat(c[pool], np.diff(pool_ptr)) * pool_w)
    Pm = sp.csr_matrix((pp, pool_tgt, pool_ptr), shape=(POOL, n))
    val = 1.0 + h; notr = np.ones(n); chosen = []; avail = np.ones(POOL, bool)
    for _ in range(K):
        gain = Pm @ (notr * val) + notr[pool]
        gain[~avail] = -1; j = int(np.argmax(gain)); avail[j] = False; chosen.append(pool[j])
        sl = slice(pool_ptr[j], pool_ptr[j + 1])
        notr[pool_tgt[sl]] *= 1 - pp[sl]; notr[pool[j]] = 0
    return np.array(chosen)

def expected(ep, seeds, rng, sims=EVAL_SIMS):
    return float(np.mean([cascade(ep, seeds, rng)[0] for _ in range(sims)]))

# ---------------------------------------------------- BO rule space (stage 1)
def zs(x):
    s = x.std(); return (x - x.mean()) / (s if s > 0 else 1)
Zd, Zp, Zc = zs(g.outdeg), zs(pagerank_endorsement(g)), zs(coreness(g))
def bo_decode(w):
    w = np.clip(w, 0, 1); return g.topk(w[0] * Zd + w[1] * Zp + w[2] * Zc)

# ---------------------------------------------------- one run of one policy
def run_policy(name, c_true, ep_true, run, fixed=None, fixed_value=None):
    rng = np.random.default_rng(10_000 * run + sum(map(ord, name)))
    a0, b0 = ALPHA, ALPHA / C                                      # agent's prior
    Ssum = np.zeros(n); Msum = np.zeros(n)
    X, y = [], []
    rewards, sets = [], []
    for t in range(T):
        if fixed is not None:
            seeds = fixed
        elif name == 'CUCB':
            ucb = gamma_dist.ppf(1 - 1.0 / (t + 2) ** 1.0, a0 + Ssum, scale=1.0 / (b0 + Msum))
            seeds = greedy(ucb)
        elif name == 'Thompson':
            seeds = greedy(rng.gamma(a0 + Ssum, 1.0 / (b0 + Msum)))
        elif name == 'BO':
            if t < 5:
                x = rng.uniform(0, 1, 3)
            else:
                gp = GaussianProcessRegressor(kernel=default_kernel(0.5)(), normalize_y=True,
                                              n_restarts_optimizer=1, random_state=run)
                gp.fit(np.array(X), np.array(y))
                cand = rng.uniform(0, 1, (800, 3))
                x = cand[np.argmax(expected_improvement(cand, gp, max(y)))]
            X.append(x); seeds = bo_decode(x)
        size, S, M = cascade(ep_true, seeds, rng)
        Ssum += S; Msum += M
        if name == 'BO': y.append(size)
        rewards.append(size); sets.append(np.asarray(seeds))
    # expected value of the final round's seed set (what the policy has converged to)
    final_expected = fixed_value if fixed is not None else expected(ep_true, sets[-1], np.random.default_rng(run + 555))
    return dict(rewards=np.array(rewards), final_seeds=sets[-1], final_expected=final_expected)

results = {}
for wname, c_true in WORLDS.items():
    print('\n=== %s ===' % wname)
    ep_true = eprob(c_true)
    t0 = time.time()
    oracle_set = greedy(c_true); prior_set = greedy(np.full(n, C)); td_set = g.topk(g.outdeg)
    r = np.random.default_rng(42)
    ref = {k: expected(ep_true, s, r, 3 * EVAL_SIMS) for k, s in
           [('oracle', oracle_set), ('prior-greedy', prior_set), ('top-degree', td_set)]}
    print('reference sets (%d sims): ' % (3 * EVAL_SIMS) + ', '.join('%s %.0f' % kv for kv in ref.items())
          + ' [%.0fs]' % (time.time() - t0))
    # check the fast oracle against snapshot CELF under the true model
    g.eprob = ep_true
    celf_set, _ = celf_snapshot(g, pool, n_worlds=30 if QUICK else 100)
    g.eprob = None
    ref['celf'] = expected(ep_true, celf_set, r, 3 * EVAL_SIMS)
    print('  CELF (snapshot, true model) %.0f; oracle %.0f; oracle/CELF %.4f; oracle&top-degree share %d/%d'
          % (ref['celf'], ref['oracle'], ref['oracle'] / ref['celf'], len(set(oracle_set) & set(td_set)), K))
    W = dict(ref=ref, c=c_true[pool], oracle_set=oracle_set, prior_set=prior_set, td_set=td_set, policies={})
    fixed = {'oracle': oracle_set, 'top-degree': td_set, 'prior-greedy': prior_set}
    for pname in POLICIES:
        t0 = time.time()
        runs = [run_policy(pname, c_true, ep_true, k, fixed.get(pname), ref.get(pname)) for k in range(N_RUNS)]
        R = np.array([x['rewards'] for x in runs])
        W['policies'][pname] = dict(rewards=R, final_expected=np.array([x['final_expected'] for x in runs]),
                                    final_seeds=[x['final_seeds'] for x in runs])
        print('  %-12s mean reward/round %.0f (first 10: %.0f, last 10: %.0f); final set E=%.0f  [%.0fs]'
              % (pname, R.mean(), R[:, :10].mean(), R[:, -10:].mean(),
                 W['policies'][pname]['final_expected'].mean(), time.time() - t0))
    results[wname] = W

out = dict(results=results, T=T, N_RUNS=N_RUNS, K=K, F=F, C=C, ALPHA=ALPHA, POOL=POOL, EVAL_SIMS=EVAL_SIMS, QUICK=QUICK)
pickle.dump(out, open(os.path.join(HERE, 'unknown_dynamics%s%s.pkl' % ('_gamma' if GAMMAS else '', '_quick' if QUICK else '')), 'wb'))
print('saved; total %.0fs' % (time.time() - t_start))
