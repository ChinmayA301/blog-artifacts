"""Stage 1 figures (rule tuning). All values read from pipeline/bo_results.pkl."""
from style import *

R = load(1)
B, H = R['baselines'], R['heur']

# ---- FIG 1: convergence, matched evaluation axis ----
fig, ax = plt.subplots(figsize=(11, 6))
convergence(ax, R)
ax.axhline(B['rand_mean'], color=MUTE, ls=':', lw=1.6, label='Random seeds')
ax.set_title('Stage 1: BO vs random search over a 3-D centrality blend', loc='left')
ax.legend(loc='lower right', frameon=True, fontsize=9); clean(ax)
caption(ax, setup_line(R))
save('bo_convergence.png')

# ---- FIG 2: strategy comparison ----
s = R['bo_summary']
labels = ['Random\nseeds', 'Top-k-core', 'Top-PageRank', 'Top-degree', 'BO (mean of %d runs)' % s['n_runs']]
vals = [H['Random seeds'], H['Top-k-core'], H['Top-PageRank'], H['Top-degree'], s['mean']]
errs = [B['rand_std'], 0, 0, 0, s['std']]
cols = [MUTE, PURPLE, ORANGE, GREEN, BLUE]
order = np.argsort(vals)
fig, ax = plt.subplots(figsize=(11, 6))
yp = np.arange(len(labels))
bars = ax.barh(yp, [vals[i] for i in order], xerr=[errs[i] for i in order],
               color=[cols[i] for i in order], edgecolor='white', alpha=0.92,
               error_kw=dict(ecolor='#444', capsize=4, lw=1))
ax.set_yticks(yp); ax.set_yticklabels([labels[i] for i in order])
for b, i in zip(bars, order):
    ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2, f'{vals[i]:.0f}', va='center', fontsize=10)
ax.set_xlabel('Expected influence spread (nodes activated, %d fresh IC sims)' % R['FINAL_SIMS'])
ax.set_title('Stage 1: seeding strategies on the same objective', loc='left')
ax.set_xlim(0, max(vals) * 1.15); clean(ax)
caption(ax, 'BO bar = mean ± sd of each run’s best weights, re-evaluated with fresh simulations '
            '(%d/%d runs above top-degree).\nBest single BO seed set shares %d/%d seeds with top-degree. '
            'PageRank is computed on the endorsement graph (retweeter → retweeted).'
        % (s['n_beat_td'], s['n_runs'], R['shared'], R['K']))
save('bo_seeding_comparison.png')

# ---- FIG 3: learned weights ----
w = np.clip(R['best_w'], 0, 1); wv = w / w.sum() if w.sum() else w
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(['Out-degree\n(%s)' % DEG_LABEL, 'PageRank', 'k-core'], wv,
              color=[GREEN, ORANGE, PURPLE], edgecolor='white', alpha=0.9, width=0.55)
for b, v in zip(bars, wv):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.0%}', ha='center', va='bottom', fontsize=11)
ax.set_ylabel('Normalized weight in best BO blend')
ax.set_title('Stage 1: weights of the best BO run', loc='left')
ax.set_ylim(0, max(wv) * 1.25 if max(wv) else 1); clean(ax)
caption(ax, 'Weights of the single best run (by fresh re-evaluation). Many weight vectors map to the same top-%d set, '
            'so these are one point\nin a flat region, not a unique optimum.' % R['K'], y=-0.13)
save('bo_learned_weights.png')
print('weights:', np.round(wv, 3))
