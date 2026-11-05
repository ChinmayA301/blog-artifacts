"""Stage 2 figures. All values read from pipeline/bo_strong_results.pkl."""
from style import *

R = load(2)
B, s, r = R['baselines'], R['bo_summary'], R['rs_summary']

# ---- FIG 1: convergence ----
fig, ax = plt.subplots(figsize=(11, 6))
convergence(ax, R)
ax.axhline(R['celf_final'], color=PURPLE, ls='-.', lw=1.6, label='CELF (%.0f)' % R['celf_final'])
ax.set_title('Stage 2: BO vs random search, 4-knob rule space with redundancy penalty', loc='left')
ax.legend(loc='lower right', frameon=True, fontsize=9); clean(ax)
caption(ax, setup_line(R))
save('bo_strong_convergence.png')

# ---- FIG 2: comparison, run-level spread ----
labels = ['Random\nseeds', 'CELF', 'Top-degree', 'Random search\n(mean of runs)', 'BO\n(mean of runs)']
means = [B['rand_mean'], R['celf_final'], B['td_final'], r['mean'], s['mean']]
errs = [B['rand_std'], R['celf_test']['std'], B['td_std'], r['std'], s['std']]
cols = [MUTE, PURPLE, GREEN, ORANGE, BLUE]
fig, ax = plt.subplots(figsize=(11, 6))
yp = np.arange(len(labels))
bars = ax.barh(yp, means, xerr=errs, color=cols, edgecolor='white', alpha=0.92,
               error_kw=dict(ecolor='#444', capsize=4, lw=1.2))
ax.set_yticks(yp); ax.set_yticklabels(labels)
for b, m in zip(bars, means):
    ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2, f'{m:.0f}', va='center', fontsize=10)
ax.set_xlabel('Expected influence spread (nodes activated)')
ax.set_title('Stage 2: BO %+.1f%% vs top-degree (mean over runs)' % pct(s['mean'], B['td_final']), loc='left')
ax.set_xlim(0, max(means) * 1.18); clean(ax)
caption(ax, 'BO / random-search bars: mean ± sd across runs of each run’s best, re-evaluated fresh '
            '(BO: %d/%d runs above top-degree).\nBest BO seed set vs top-degree, set-level Welch test: t=%.2f, p=%.2g '
            '(compares two fixed seed sets, not the method). CELF: lazy greedy on 200 live-edge snapshots.'
        % (s['n_beat_td'], s['n_runs'], R['setlevel']['t'], R['setlevel']['p']))
save('bo_strong_comparison.png')

# ---- FIG 3: mechanism - overlap with top-degree ----
fig, ax = plt.subplots(figsize=(11, 6))
cats = ['Shared with\ntop-degree', 'Different from\ntop-degree']
vals = [R['shared'], R['K'] - R['shared']]
bars = ax.bar(cats, vals, color=[GREEN, BLUE], edgecolor='white', width=0.5, alpha=0.9)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, str(v), ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Seeds (of k=%d)' % R['K']); ax.set_ylim(0, R['K'] + 2)
ax.set_title('Stage 2: how different is the best BO seed set?', loc='left'); clean(ax)
caption(ax, 'Direct-audience overlap (share of seeds’ retweeters reached twice): top-degree %.2f, best BO %.2f, CELF %.2f.\n'
            'CELF shares %d/%d seeds with top-degree.'
        % (R['ov_td'], R['ov_bo'], R['ov_celf'], R['celf_shared'], R['K']), y=-0.13)
save('bo_strong_mechanism.png')
