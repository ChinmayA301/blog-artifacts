"""Does anything beat top-degree once the simulator is checked against real data?

Compares CELF and the best BO stage against top-degree under three simulators,
read from the generated summaries in ../results/:
  1. retweet graph, uniform IC p=0.12            (the original study's substrate)
  2. follower graph, weighted cascade, no floor  (fails the per-account check: node 1)
  3. follower graph, weighted cascade, floor 20  (floor fixed by out-of-sample fit)
"""
import json
from style import *

RES = os.path.join(HERE, '..', 'results')
settings = [('summary.json', 'Retweet graph\nuniform IC p=0.12'),
            ('summary_follower_wc.json', 'Follower graph\nweighted cascade, no floor'),
            ('summary_follower_wc_F20.json', 'Follower graph\nweighted cascade, floor 20\n(best out-of-sample fit)')]
rows = []
for f, lab in settings:
    s = json.load(open(os.path.join(RES, f)))
    td = s['baselines']['top_degree']
    best_stage = max(s['stages'], key=lambda k: s['stages'][k]['bo']['mean'])
    rows.append((lab, pct(s['controls']['celf_final'], td), pct(s['stages'][best_stage]['bo']['mean'], td), best_stage))

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(rows)); w = 0.34
b1 = ax.bar(x - w / 2, [r[1] for r in rows], w, color=PURPLE, alpha=0.9, label='CELF (near-optimal greedy)')
b2 = ax.bar(x + w / 2, [r[2] for r in rows], w, color=BLUE, alpha=0.9, label='Best BO stage (mean of runs)')
for bars in (b1, b2):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 1, f'{v:+.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.axhline(0, color=GREEN, lw=1.6, ls='--', label='Top-degree')
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
ax.set_ylabel('Spread vs top-degree (%)')
ax.set_title('Beating top-degree depends on the simulator; the best-checked one says no', loc='left')
ax.legend(loc='upper right', frameon=True, fontsize=9); clean(ax)
ax.annotate('driven by a crawl artifact:\none account (node 1) credited with\na ~12k cascade; observed retweets: 0',
            xy=(1, rows[1][1]), xytext=(1.35, rows[1][1] * 0.55), fontsize=8.5, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED, lw=1))
caption(ax, 'k=20 seeds, Higgs data. Floor 20 = p_uv = c / max(followees(v), 20), the floor with the best out-of-sample fit to real diffusion '
            '(calibrated 1–3 July,\ntested from 4 July), fixed before the optimisation was run. Best BO stage: '
            + ', '.join('stage %s' % r[3] for r in rows) + ' respectively.')
save('substrate_comparison.png')
print([(r[0].split(chr(10))[0], round(r[1], 2), round(r[2], 2)) for r in rows])
