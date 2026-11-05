"""Cross-stage + free-node figures. All values read from the four stage pickles.
Every stage uses the same objective (k, IC p, MC) and the same top-degree
baseline (pipeline/baselines.pkl); this script asserts that."""
from style import *

S = {i: load(i) for i in (1, 2, 3, 4)}
for k in ('K', 'IC_P', 'MC'):
    assert len({S[i][k] for i in S}) == 1, 'stages disagree on %s' % k
assert len({S[i]['baselines']['td_final'] for i in S}) == 1, 'stages use different top-degree baselines'
B = S[1]['baselines']; TD = B['td_final']

# ---- FIG 1: four design spaces, one objective ----
fig, ax = plt.subplots(figsize=(12, 6.2))
names = ['1. Rule-tuning\n(%d-D)' % S[1]['dim'], '2. Rules + redundancy\n(%d-D)' % S[2]['dim'],
         '3. Free-node, naive\n(%d-D)' % S[3]['dim'], '4. Free-node, constrained\n(%d-D)' % S[4]['dim']]
x = np.arange(4); w = 0.26
bo_m = [S[i]['bo_summary']['mean'] for i in S]; bo_s = [S[i]['bo_summary']['std'] for i in S]
rs_m = [S[i]['rs_summary']['mean'] for i in S]; rs_s = [S[i]['rs_summary']['std'] for i in S]
ax.bar(x - w, [TD] * 4, w, color=GREEN, alpha=0.85, label='Top-degree (shared)', edgecolor='white')
ax.bar(x, rs_m, w, yerr=rs_s, color=ORANGE, alpha=0.85, label='Random search, same space & budget',
       edgecolor='white', error_kw=dict(ecolor='#444', capsize=3))
b2 = ax.bar(x + w, bo_m, w, yerr=bo_s, color=BLUE, alpha=0.9, label='BO', edgecolor='white',
            error_kw=dict(ecolor='#444', capsize=3))
for i, b in zip(S, b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02 + bo_s[i - 1], '%+.1f%%' % pct(bo_m[i - 1], TD),
            ha='center', fontsize=9, color=BLUE, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel('Expected influence spread (nodes activated)')
ax.set_title('Same objective, four search spaces: BO vs random search vs top-degree', loc='left')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=9, ncol=3)
ax.set_ylim(0, max(bo_m + [TD]) * 1.15); clean(ax)
caption(ax, 'k=%d, IC p=%s, %d sims/evaluation, all stages. Bars = mean ± sd across runs of each run’s best, '
            're-evaluated with %d fresh sims. %% = BO mean vs top-degree.'
        % (S[1]['K'], S[1]['IC_P'], S[1]['MC'], S[1]['FINAL_SIMS']), y=-0.27)
save('bo_freenode_progression.png')

# ---- FIG 2: stage 3 convergence ----
R = S[3]
fig, ax = plt.subplots(figsize=(11, 6))
convergence(ax, R, bo_label='Free-node BO (naive)')
ax.set_title('Stage 3: naive free-node search in %d dimensions' % R['dim'], loc='left')
ax.legend(loc='center right', frameon=True, fontsize=9); clean(ax)
caption(ax, '%d-D search (%dD spectral embedding × %d anchors) over %d embedded nodes (%d/%d top-degree seeds are '
            'in that pool).\n' % (R['dim'], R['D'], R['K'], R['pool'], R['td_in_pool'], R['K']) + setup_line(R))
save('bo_freenode_failure.png')

# ---- FIG 3: stage 4 attribution - is it BO or the decoder? ----
R = S[4]
fig, ax = plt.subplots(figsize=(11, 6))
rows = [('Top-degree', [TD], GREEN), ('Decoder only\n(no BO, λ grid)', [R['dec_final']], PURPLE),
        ('Random search\n(13-D, same budget)', list(R['rs_finals']), ORANGE), ('BO (13-D)', list(R['bo_finals']), BLUE)]
for yi, (lab, v, col) in enumerate(rows):
    ax.scatter(v, [yi] * len(v), s=70, color=col, zorder=3, edgecolors='white')
    ax.plot([np.mean(v)] * 2, [yi - 0.25, yi + 0.25], color=col, lw=2.5)
ax.axvline(TD, color=GREEN, ls='--', lw=1)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel('Expected influence spread (fresh %d-sim re-evaluation)' % R['FINAL_SIMS'])
ax.set_title('Stage 4: one dot per run — does the gain belong to BO or to the decoder?', loc='left'); clean(ax)
caption(ax, 'BO runs above top-degree: %d/%d; above decoder-only: %d/%d. Vertical tick = mean across runs.\n'
            'Decoder-only seed set shares %d/%d seeds with top-degree, so any gap between those two is Monte Carlo noise.'
        % (R['bo_summary']['n_beat_td'], R['bo_summary']['n_runs'], int((R['bo_finals'] > R['dec_final']).sum()),
           R['bo_summary']['n_runs'], R['dec_shared'], R['K']), y=-0.13)
save('bo_freenode_synthesis.png')
