"""
calibrate_and_holdout.py - calibrate cascade models on the Higgs follower graph
from the real activity log, then test them on a later, held-out time window.

Data (SNAP Higgs, place in ../pipeline/):
  follower_graph.npz           built by ../pipeline/00b_build_follower_graph.py
  higgs-activity_time.txt.gz   "A B timestamp TYPE"; TYPE RT means A retweeted B

Models (edge u -> v means v follows u, so v is exposed to u):
  uniform    p_uv = p                                 one probability for every edge
  wc (F)     p_uv = c / max(followees(v), F)          weighted cascade: attention is split
                                                      across everyone v follows; F floors the
                                                      denominator (see below)
Calibration window : events before 2012-07-04 00:00 UTC (T_SPLIT)
Held-out window    : events from T_SPLIT on (includes the 4 July announcement)

Calibration counts, over every user active in the window, how many followers
retweeted them (follow-explained adoptions) against the model's expected
exposure mass. Restricting to users who were retweeted at least once would
bias the estimate upward, so every active user counts.

Why the floor F: the SNAP crawl records follows only among Higgs-involved
users, so followee counts are truncated. 33k users appear to follow exactly one
account, and half of them follow node 1, which was never retweeted. With F = 1,
weighted cascade gives each of those users p ~ 0.49 and credits node 1 with a
~12,000-node cascade. F is chosen by fit (spread and
depth) in the out-of-sample test, and node 1's predicted reach is reported as a sanity check.
Aggregate fit cannot see this artifact (node 1 is never an active seed), so see also
account_ranking_check.py, which tests the per-account ranking seed selection depends on.

Held-out test: seed each model with the window's originators (users retweeted
via a follow edge who did not themselves retweet anyone in that window),
simulate, and compare with what happened:
  - number of follow-explained adopters
  - depth: share of adopters who were themselves retweeted by a follower
  - one-hop calibration: predicted vs observed direct adoptions per seed

Outputs: calibration.json, holdout_results.json, holdout_check.png (this folder).
"""
import os, gzip, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, '..', 'pipeline')
T_SPLIT = 1341360000          # 2012-07-04 00:00:00 UTC
R_SIMS = 20
FLOORS = [1, 5, 10, 20, 50]
rng = np.random.default_rng(2026)

G = np.load(os.path.join(PIPE, 'follower_graph.npz'))
indptr, indices, ids = G['indptr'], G['indices'], G['ids']
n, m = len(ids), len(indices)
followers = np.diff(indptr)                       # out-degree: audience size
followees = np.bincount(indices, minlength=n)     # in-degree: accounts v follows (truncated by the crawl)
pos = np.full(ids.max() + 1, -1, dtype=np.int64); pos[ids] = np.arange(n)
src_of_edge = np.repeat(np.arange(n), followers)
edge_key_sorted = np.sort(src_of_edge * n + indices)
def is_edge(u, v):
    k = u * n + v
    i = np.searchsorted(edge_key_sorted, k); i[i >= m] = 0
    return edge_key_sorted[i] == k

def row_sum(w):
    return np.bincount(src_of_edge, weights=w, minlength=n)

# ---- activity log ----
raw = [l.split() for l in gzip.open(os.path.join(PIPE, 'higgs-activity_time.txt.gz'), 'rt')]
a = np.array([int(r[0]) for r in raw]); b = np.array([int(r[1]) for r in raw])
ts = np.array([int(r[2]) for r in raw]); kind = np.array([r[3] for r in raw])
ok = (a <= ids.max()) & (b <= ids.max())
a, b, ts, kind = pos[a[ok]], pos[b[ok]], ts[ok], kind[ok]
ok = (a >= 0) & (b >= 0)
a, b, ts, kind = a[ok], b[ok], ts[ok], kind[ok]
observed_rt = np.bincount(b[kind == 'RT'], minlength=n)

def window(mask):
    act = np.unique(np.concatenate([a[mask], b[mask]]))
    rt = mask & (kind == 'RT')
    pairs = np.unique(np.stack([b[rt], a[rt]], 1), axis=0)          # (source, retweeter)
    fol = is_edge(pairs[:, 0], pairs[:, 1])                          # retweeter follows source
    return act, pairs, pairs[fol]

wc_mass = {F: row_sum(1.0 / np.maximum(followees, F)[indices]) for F in FLOORS}
mean_followers_reached = followers[indices].mean()

def calibrate(mask, label):
    act, pairs, fpairs = window(mask)
    adoptions = int(np.isin(fpairs[:, 0], act).sum())
    p = adoptions / followers[act].sum()
    return dict(window=label, active_users=int(len(act)), retweet_pairs=int(len(pairs)),
                follow_explained_share=float(len(fpairs) / len(pairs)), follow_explained_adoptions=adoptions,
                exposures=int(followers[act].sum()), uniform_p=float(p),
                uniform_reproduction_number=float(p * mean_followers_reached),
                wc_c_by_floor={str(F): float(adoptions / wc_mass[F][act].sum()) for F in FLOORS})

masks = dict(pre=ts < T_SPLIT, in_window=ts >= T_SPLIT, full=np.ones(len(ts), bool))
labels = dict(pre='before 2012-07-04', in_window='from 2012-07-04 (held-out window itself)', full='whole week')
cal = dict(t_split_utc='2012-07-04T00:00:00Z', **{k: calibrate(v, labels[k]) for k, v in masks.items()})

# ---- held-out window ----
act_te, pairs_te, fpairs_te = window(masks['in_window'])
seeds = np.setdiff1d(np.unique(fpairs_te[:, 0]), np.unique(pairs_te[:, 1]))
obs_adopters = np.setdiff1d(np.unique(fpairs_te[:, 1]), seeds)
obs_depth = float(np.isin(obs_adopters, fpairs_te[:, 0]).mean())
direct_obs = np.bincount(fpairs_te[:, 0], minlength=n)[seeds]
print('held-out: %d seeds (originators), %d observed follow-explained adopters, depth %.3f'
      % (len(seeds), len(obs_adopters), obs_depth))

def edge_probs(model, prm, F=1):
    if model == 'uniform':
        return np.full(m, prm['uniform_p'])
    return np.minimum(1.0, prm['wc_c_by_floor'][str(F)] / np.maximum(followees, F)[indices])

def simulate(eprob, seeds):
    active = np.zeros(n, bool); infected_someone = np.zeros(n, bool)
    active[seeds] = True; touched = [seeds]; frontier = seeds
    while frontier.size:
        starts = indptr[frontier]; lens = followers[frontier]; tot = int(lens.sum())
        if tot == 0: break
        offs = np.repeat(starts - (np.cumsum(lens) - lens), lens) + np.arange(tot)
        hit = rng.random(tot) < eprob[offs]
        tgt = indices[offs[hit]]; par = np.repeat(frontier, lens)[hit]
        new = ~active[tgt]; tgt, par = tgt[new], par[new]
        tgt, first = np.unique(tgt, return_index=True); par = par[first]
        active[tgt] = True; infected_someone[par] = True
        touched.append(tgt); frontier = tgt
    allt = np.concatenate(touched); nonseed = allt[~np.isin(allt, seeds)]
    return len(nonseed), float(infected_someone[nonseed].mean()) if len(nonseed) else 0.0

res = dict(observed=dict(seeds=int(len(seeds)), adopters=int(len(obs_adopters)), depth=obs_depth))
bins = [0, 10, 100, 1000, 10000, 10**7]
bin_idx = np.digitize(followers[seeds], bins) - 1
node1 = int(pos[1])

def evaluate(key, ep):
    t0 = time.time()
    sims = [simulate(ep, seeds) for _ in range(R_SIMS)]
    sz = np.array([s[0] for s in sims]); dp = np.array([s[1] for s in sims])
    direct_pred = row_sum(ep)[seeds]
    calib = [dict(followers_bin='%d-%d' % (bins[k], bins[k + 1] - 1), seeds=int((bin_idx == k).sum()),
                  predicted_direct=float(direct_pred[bin_idx == k].sum()), observed_direct=int(direct_obs[bin_idx == k].sum()))
             for k in range(len(bins) - 1) if (bin_idx == k).any()]
    err = abs(np.log(sz.mean() / len(obs_adopters))) + abs(np.log(dp.mean() / obs_depth))
    res[key] = dict(adopters_mean=float(sz.mean()), adopters_p5=float(np.percentile(sz, 5)),
                    adopters_p95=float(np.percentile(sz, 95)), depth_mean=float(dp.mean()),
                    ratio_predicted_to_observed=float(sz.mean() / len(obs_adopters)),
                    fit_error_log=float(err), one_hop_calibration=calib)
    print('%-22s predicted adopters %7.0f [%.0f, %.0f] vs observed %d (ratio %.2f); depth %.3f vs %.3f; fit error %.2f  (%.0fs)'
          % (key, sz.mean(), *np.percentile(sz, [5, 95]), len(obs_adopters), sz.mean() / len(obs_adopters),
             dp.mean(), obs_depth, err, time.time() - t0))

for params in ('pre', 'in_window'):
    evaluate('uniform_' + params, edge_probs('uniform', cal[params]))
    for F in FLOORS:
        evaluate('wc_F%d_%s' % (F, params), edge_probs('wc', cal[params], F))

# ---- node-1 sanity check under whole-week parameters ----
node1_check = {}
for F in FLOORS:
    ep = edge_probs('wc', cal['full'], F)
    node1_check[str(F)] = float(np.mean([simulate(ep, np.array([node1]))[0] for _ in range(R_SIMS)]))
node1_check['uniform'] = float(np.mean([simulate(edge_probs('uniform', cal['full']), np.array([node1]))[0] for _ in range(R_SIMS)]))
res['node1_sanity'] = dict(followers=int(followers[node1]), observed_retweets=int(observed_rt[node1]),
                           predicted_singleton_reach=node1_check)
print('node 1: %d followers, observed retweets %d, predicted reach by floor: %s'
      % (followers[node1], observed_rt[node1], {k: round(v) for k, v in node1_check.items()}))

best_F = min(FLOORS, key=lambda F: res['wc_F%d_pre' % F]['fit_error_log'])
cal['chosen'] = dict(model='wc', floor=best_F, c=cal['full']['wc_c_by_floor'][str(best_F)],
                     rule='floor with the lowest OUT-OF-SAMPLE fit error (calibrated before 4 July, tested from '
                          '4 July; |log spread ratio| + |log depth ratio|), fixed before any optimisation was run on it; '
                          'c from the whole-week calibration at that floor')
print('chosen: weighted cascade, floor F=%d, c=%.4f' % (best_F, cal['chosen']['c']))

json.dump(cal, open(os.path.join(HERE, 'calibration.json'), 'w'), indent=2)
json.dump(res, open(os.path.join(HERE, 'holdout_results.json'), 'w'), indent=2)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
keys = ['uniform_pre', 'wc_F%d_pre' % best_F, 'uniform_in_window', 'wc_F1_in_window', 'wc_F%d_in_window' % best_F]
labels_ = ['Observed', 'Uniform\n(pre)', 'Weighted F=%d\n(pre)' % best_F, 'Uniform\n(in-window)',
           'Weighted F=1\n(in-window)', 'Weighted F=%d\n(in-window)' % best_F]
cols = ['#55A868', '#E8A0A0', '#9DB4D8', '#C44E52', '#B0B0B0', '#4C72B0']
vals = [len(obs_adopters)] + [res[k]['adopters_mean'] for k in keys]
dvals = [obs_depth] + [res[k]['depth_mean'] for k in keys]
for x, v, fmt, yl, tt in [(ax[0], vals, '{:,.0f}', 'Follow-explained adopters (held-out window)', 'Total spread'),
                          (ax[1], dvals, '{:.3f}', 'Share of adopters who were themselves retweeted', 'Cascade depth')]:
    x.bar(labels_, v, color=cols)
    for i, y in enumerate(v): x.text(i, y, fmt.format(y), ha='center', va='bottom', fontsize=8.5)
    x.set_ylabel(yl); x.set_title(tt, loc='left'); x.tick_params(axis='x', labelsize=8)
    x.spines['top'].set_visible(False); x.spines['right'].set_visible(False)
fig.suptitle('Held-out check from 4 July 2012: "pre" = calibrated on 1-3 July; "in-window" = calibrated on the test window',
             x=0.01, ha='left', fontweight='bold', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(HERE, 'holdout_check.png'), dpi=150, bbox_inches='tight')
print('saved calibration.json, holdout_results.json, holdout_check.png')
