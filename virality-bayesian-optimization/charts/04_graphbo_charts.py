"""
Companion visuals ("BO meets graphs").
  1. Node scatter of the high-degree subgraph: where the top-degree seeds sit
     vs where the best stage-4 BO seed set sits (read from its pickle).
  2. Combinatorial blow-up: number of possible seed sets vs k (computed).
"""
import sys
from math import comb, lgamma, log
import networkx as nx
from style import *

sys.path.insert(0, PIPE)
from common import load_graph  # noqa: E402  (asserts the corrected edge direction)

g = load_graph()
R4 = load(4)
POOL = 800
pool = g.topk(g.outdeg, POOL)
sub = g.H.subgraph(g.nodes[pool]).to_undirected()
cc = max(nx.connected_components(sub), key=len)
sub_cc = sub.subgraph(cc)
emb_ids = np.array(sub_cc.nodes()); emb_ix = np.array([g.idx[n] for n in emb_ids])
edeg = g.outdeg[emb_ix]
pos = nx.spring_layout(sub_cc, k=0.15, iterations=40, seed=1)
P = np.array([pos[n] for n in emb_ids]); P = (P - P.min(0)) / (np.ptp(P, axis=0) + 1e-9)
where = {ix: i for i, ix in enumerate(emb_ix)}

fig, ax = plt.subplots(figsize=(11, 7.5))
sc = ax.scatter(P[:, 0], P[:, 1], s=20 + 380 * (edeg / edeg.max()), c=np.log1p(edeg), cmap='viridis',
                alpha=0.75, edgecolors='white', linewidths=0.3, zorder=2)
def mark(seeds, **kw):
    ii = [where[s] for s in seeds if s in where]
    ax.scatter(P[ii, 0], P[ii, 1], zorder=4, **kw)
    return len(ii)
n_td = mark(R4['baselines']['td_seeds'], s=200, marker='o', facecolors='none', edgecolors=RED, linewidths=2.4,
            label='Top-degree seeds')
n_bo = mark(R4['best_seeds'], s=150, marker='D', facecolors='none', edgecolors='#111', linewidths=2.0,
            label='Best stage-4 BO seeds')
plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02).set_label('log out-degree (# retweeters)', fontsize=9)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('Where the strategies seed in the high-degree subgraph', loc='left')
ax.legend(loc='upper right', frameon=True, fontsize=9)
caption(ax, 'Largest connected component (%d nodes) of the top-%d out-degree subgraph, force-directed layout. '
            'Shown: %d/%d top-degree seeds,\n%d/%d BO seeds (the rest fall outside this component). '
            'Best BO set re-evaluated at %.0f vs top-degree %.0f.'
        % (len(emb_ix), POOL, n_td, R4['K'], n_bo, R4['K'], R4['best_final'], R4['baselines']['td_final']), y=-0.04)
save('graphbo_acquisition_landscape.png')

# ---- FIG 2: combinatorial blow-up ----
def log10comb(n, k):
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / log(10)
ks = np.arange(1, 31)
fig, ax = plt.subplots(figsize=(11, 6))
for N, col, lab in [(POOL, ORANGE, 'candidate pool (N=%d)' % POOL), (g.N, BLUE, 'full graph (N=%s)' % f'{g.N:,}')]:
    ax.semilogy(ks, [comb(N, int(k)) for k in ks], lw=2.4, color=col, marker='o', ms=3, label=lab)
ax.axhline(1e6, color=MUTE, ls=':', lw=1.4); ax.text(1, 1.5e6, 'a million evaluations', fontsize=8.5, color=MUTE)
K = R4['K']
ax.axvline(K, color=RED, ls='--', lw=1.6); ax.text(K + 0.3, 1e30, 'k = %d\n(our budget)' % K, color=RED, fontsize=9)
ax.set_xlabel('Seed set size k'); ax.set_ylabel('Number of possible seed sets (log scale)')
ax.set_title('Why you cannot brute-force seed selection', loc='left')
ax.legend(loc='lower right', frameon=True, fontsize=9); clean(ax)
full, small = log10comb(g.N, K), log10comb(POOL, K)
caption(ax, 'At k=%d the full graph admits ~10^%d seed sets; even a %d-node candidate pool gives ~10^%d.\n'
            'BO therefore searches a low-dimensional parameterization of seed sets, never the sets themselves.'
        % (K, int(full), POOL, int(small)))
save('graphbo_combinatorial.png')
print('C(%d,%d) ~ 10^%.1f ; C(%d,%d) ~ 10^%.1f' % (g.N, K, full, POOL, K, small))
