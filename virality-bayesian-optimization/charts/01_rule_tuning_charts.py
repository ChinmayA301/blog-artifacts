import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os, numpy as np, pickle

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi':150,'font.size':11,'axes.titlesize':14,
                     'axes.titleweight':'bold','axes.labelsize':11,'savefig.dpi':150})
BLUE='#4C72B0'; ORANGE='#DD8452'; GREEN='#55A868'; RED='#C44E52'; PURPLE='#8172B3'; MUTE='#9AA0A6'
OUT='figures'
os.makedirs(OUT, exist_ok=True)

R = pickle.load(open('/home/claude/bo_results.pkl','rb'))
bo, rs = R['bo_curves'], R['rand_curves']
base = R['base_results']

# ---------- FIG 1: convergence trace, BO vs random search ----------
fig, ax = plt.subplots(figsize=(11,6))
x = np.arange(1, bo.shape[1]+1)
for curves, col, lab in [(bo,BLUE,'Bayesian Optimization'),(rs,ORANGE,'Random search')]:
    m = curves.mean(0); lo=curves.min(0); hi=curves.max(0)
    ax.plot(x, m, color=col, lw=2.4, label=lab, marker='o', ms=3)
    ax.fill_between(x, lo, hi, color=col, alpha=0.16)

ax.axhline(base['Top-degree'], color=GREEN, ls='--', lw=1.6, label='Top-degree heuristic')
ax.axhline(base['Random seeds'], color=MUTE, ls=':', lw=1.6, label='Random seeds')
ax.set_xlabel('Objective evaluations (IC simulations)')
ax.set_ylabel('Best expected spread found (nodes activated)')
ax.set_title('BO finds high-spread seeds in fewer evaluations than random search', loc='left')
ax.legend(loc='lower right', frameon=True, fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.text(0.0,-0.15,
        f'Higgs retweet network, largest WCC ({R["graph_nodes"]:,} nodes / {R["graph_edges"]:,} edges). '
        f'Seed budget k={R["K"]}, IC activation p={R["IC_P"]}. Bands span min\u2013max over 5 runs.',
        transform=ax.transAxes, fontsize=8.5, color=MUTE, va='top')
plt.tight_layout(); plt.savefig(f'{OUT}/bo_convergence.png', bbox_inches='tight'); plt.close()
print('fig1 done')

# ---------- FIG 2: seeding strategy comparison ----------
labels = ['Random\nseeds','Top-k-core','Top-PageRank','Top-degree','BO-optimized']
vals   = [base['Random seeds'], base['Top-k-core'], base['Top-PageRank'],
          base['Top-degree'], R['bo_final']]
errs   = [base['_random_std'],0,0,0,0]
colors = [MUTE, PURPLE, ORANGE, GREEN, BLUE]
order  = np.argsort(vals)
fig, ax = plt.subplots(figsize=(11,6))
ypos = np.arange(len(labels))
bars = ax.barh(ypos, [vals[i] for i in order], xerr=[errs[i] for i in order],
               color=[colors[i] for i in order], edgecolor='white', alpha=0.92,
               error_kw=dict(ecolor=MUTE, capsize=4, lw=1))
ax.set_yticks(ypos); ax.set_yticklabels([labels[i] for i in order])
for b,i in zip(bars, order):
    ax.text(b.get_width()+1, b.get_y()+b.get_height()/2, f'{vals[i]:.0f}',
            va='center', ha='left', fontsize=10)
ax.set_xlabel('Expected influence spread (nodes activated, IC model)')
ax.set_title('Seeding strategies compared on real diffusion spread', loc='left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_xlim(0, max(vals)*1.18)
ax.text(0.0,-0.15,
        'Top-degree and BO-optimized cluster at the top: on this graph BO rediscovers that out-degree dominates spread, '
        'rather than beating it.\nThe honest finding \u2014 BO matches the best heuristic and confirms which signal matters, at the cost of more evaluations.',
        transform=ax.transAxes, fontsize=8.5, color=MUTE, va='top')
plt.tight_layout(); plt.savefig(f'{OUT}/bo_seeding_comparison.png', bbox_inches='tight'); plt.close()
print('fig2 done')

# ---------- FIG 3: what BO learned (weight space) ----------
w = R['bo_best_w']
fig, ax = plt.subplots(figsize=(11,6))
comp = ['Out-degree','PageRank','k-core']
wv = np.clip(w,0,1); wv = wv/wv.sum()
bars = ax.bar(comp, wv, color=[GREEN,ORANGE,PURPLE], edgecolor='white', alpha=0.9, width=0.55)
for b,v in zip(bars,wv):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.0%}', ha='center', va='bottom', fontsize=11)
ax.set_ylabel('Normalized weight in BO-optimal blend')
ax.set_title('What BO learned: out-degree carries the diffusion signal', loc='left')
ax.set_ylim(0, max(wv)*1.25)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.text(0.0,-0.13,
        'BO\u2019s best seed-selection rule weights nodes almost entirely by out-degree, with a minor k-core contribution and '
        'negligible PageRank \u2014\na compact, interpretable policy recovered from a noisy black-box objective.',
        transform=ax.transAxes, fontsize=8.5, color=MUTE, va='top')
plt.tight_layout(); plt.savefig(f'{OUT}/bo_learned_weights.png', bbox_inches='tight'); plt.close()
print('fig3 done | weights:', np.round(wv,3))
