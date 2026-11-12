"""Per-account validity: does the model rank candidate seed accounts the way the
real data does? Seed selection is a ranking problem, so this is the check that
matters for influence maximization; an optimizer exploits exactly the accounts
the model overrates.

For the top-3000 accounts by followers (the CELF / stage-2 candidate pool):
  predicted = expected direct follower adoptions under the model = sum of p_uv over followers
  observed  = distinct followers who retweeted the account during the week
Reports Spearman rank correlation, top-20 overlap, and the worst over-predicted
accounts, for uniform IC and weighted cascade at each floor F."""
import os, json, gzip
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); PIPE = os.path.join(HERE, '..', 'pipeline')
G = np.load(os.path.join(PIPE, 'follower_graph.npz'))
indptr, indices, ids = G['indptr'], G['indices'], G['ids']; n, m = len(ids), len(indices)
followers = np.diff(indptr); followees = np.bincount(indices, minlength=n)
src = np.repeat(np.arange(n), followers)
pos = np.full(ids.max() + 1, -1, dtype=np.int64); pos[ids] = np.arange(n)
cal = json.load(open(os.path.join(HERE, 'calibration.json')))['full']
key = np.sort(src * n + indices)
pairs = set()
for l in gzip.open(os.path.join(PIPE, 'higgs-activity_time.txt.gz'), 'rt'):
    x, y, t, k = l.split()
    if k == 'RT' and int(x) <= ids.max() and int(y) <= ids.max():
        u, v = pos[int(y)], pos[int(x)]                    # source, retweeter
        if u >= 0 and v >= 0: pairs.add((u, v))
P = np.array(sorted(pairs)); kk = P[:, 0] * n + P[:, 1]
i = np.searchsorted(key, kk); i[i >= m] = 0; fol = key[i] == kk
observed = np.bincount(P[fol, 0], minlength=n)            # distinct follower-retweeters per account
pool = np.argsort(-followers, kind='stable')[:3000]
out = {}
models = [('uniform', np.full(m, cal['uniform_p']))] + \
         [('wc_F%s' % F, np.minimum(1.0, c / np.maximum(followees, int(F))[indices])) for F, c in cal['wc_c_by_floor'].items()]
obs_top = set(pool[np.argsort(-observed[pool])[:20]])
for name, ep in models:
    pred = np.bincount(src, weights=ep, minlength=n)
    rho = spearmanr(pred[pool], observed[pool]).correlation
    top = pool[np.argsort(-pred[pool])[:20]]
    worst = pool[np.argsort(-(np.log1p(pred[pool]) - np.log1p(observed[pool])))[:3]]
    out[name] = dict(spearman=float(rho), top20_overlap_with_observed=len(set(top) & obs_top),
                     top20_predicted_with_zero_observed=int((observed[top] == 0).sum()),
                     worst_overpredicted=[dict(node=int(ids[w]), predicted=float(pred[w]), observed=int(observed[w])) for w in worst])
    print('%-8s spearman %.3f | top-20 by model: %2d/20 in observed top-20, %d with zero observed | worst: %s'
          % (name, rho, out[name]['top20_overlap_with_observed'], out[name]['top20_predicted_with_zero_observed'],
             [(d['node'], round(d['predicted']), d['observed']) for d in out[name]['worst_overpredicted']]))
td = pool[:20]
out['top_degree'] = dict(top20_overlap_with_observed=len(set(td) & obs_top), with_zero_observed=int((observed[td] == 0).sum()))
print('top-degree (followers) top-20: %d/20 in observed top-20, %d with zero observed' % (out['top_degree']['top20_overlap_with_observed'], out['top_degree']['with_zero_observed']))
json.dump(out, open(os.path.join(HERE, 'account_ranking_check.json'), 'w'), indent=2)
