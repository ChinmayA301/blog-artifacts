import matplotlib.pyplot as plt, numpy as np, pickle
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi':150,'font.size':11,'axes.titlesize':14,
                     'axes.titleweight':'bold','axes.labelsize':11,'savefig.dpi':150})
BLUE='#4C72B0'; ORANGE='#DD8452'; GREEN='#55A868'; RED='#C44E52'; PURPLE='#8172B3'; MUTE='#9AA0A6'; TEAL='#17A2B8'
OUT='figures'
os.makedirs(OUT, exist_ok=True)
R=pickle.load(open('/home/claude/bo_freenode_results.pkl','rb'))  # naive free-node (failed)

# ---- FIG 1: the three-experiment progression (the headline figure) ----
# verified numbers across the three design spaces
fig,ax=plt.subplots(figsize=(11.5,6))
experiments=['Rule-tuning\n(reweight centralities)','Free-node, naive\n(unconstrained embedding)','Free-node, constrained\n(diverse subsets of hubs)']
bo_vals=[179.0, 91.0, 183.1]
td_vals=[174.6, 175.8, 175.2]
shared =[18, 1, 14]
x=np.arange(3); w=0.36
b1=ax.bar(x-w/2, td_vals, w, color=GREEN, alpha=0.85, label='Top-degree baseline', edgecolor='white')
b2=ax.bar(x+w/2, bo_vals, w, color=BLUE, alpha=0.9, label='BO-optimized', edgecolor='white')
for b,v in zip(b1,td_vals): ax.text(b.get_x()+b.get_width()/2,v+2,f'{v:.0f}',ha='center',fontsize=9.5)
for b,v in zip(b2,bo_vals): ax.text(b.get_x()+b.get_width()/2,v+2,f'{v:.0f}',ha='center',fontsize=9.5,color=BLUE,fontweight='bold')
# annotate shared-seed count inside the BO bars
for i,(sh,bv) in enumerate(zip(shared,bo_vals)):
    ax.text(i+w/2, bv/2, f'{sh}/20\nshared', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(experiments)
ax.set_ylabel('Expected influence spread (nodes activated, IC model)')
ax.set_title('The design space decides everything: three ways to let BO pick seeds', loc='left')
ax.legend(loc='upper center', frameon=True, fontsize=9, ncol=2)
ax.set_ylim(0,205); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.margins(y=0.1)
ax.text(0.0,-0.30,
        'Same graph, same budget, same objective. BO only beats top-degree when the search space lets diversity be expressed '
        '*among high-influence nodes*.\nUnconstrained free selection (middle) wanders into low-influence peripheral nodes and '
        'collapses to 91 \u2014 maximal diversity, minimal spread.',
        transform=ax.transAxes, fontsize=8.5, color=MUTE, va='top')
plt.tight_layout(); plt.savefig(f'{OUT}/bo_freenode_progression.png',bbox_inches='tight'); plt.close()
print('fig1 done')

# ---- FIG 2: the failure mode — diversity vs spread scatter ----
# show the naive run's convergence stalling far below baselines
bo,rs=R['bo_curves'],R['rand_curves']
fig,ax=plt.subplots(figsize=(11,6))
xx=np.arange(1,bo.shape[1]+1)
ax.plot(xx,bo.mean(0),color=BLUE,lw=2.4,marker='o',ms=3,label='Free-node BO (naive)')
ax.fill_between(xx,bo.min(0),bo.max(0),color=BLUE,alpha=0.16)
ax.plot(xx,rs.mean(0),color=ORANGE,lw=2.2,marker='s',ms=3,label='Random search')
ax.fill_between(xx,rs.min(0),rs.max(0),color=ORANGE,alpha=0.14)
ax.axhline(R['b_td'],color=GREEN,ls='--',lw=1.8,label='Top-degree (176)')
ax.axhline(183.1,color=TEAL,ls='-',lw=1.8,label='Constrained free-node BO (183)')
ax.set_xlabel('Objective evaluations (IC simulations)')
ax.set_ylabel('Best expected spread found')
ax.set_title('Why naive free selection fails: 120-D search starves at this budget', loc='left')
ax.legend(loc='center right', frameon=True, fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_ylim(60,200)
ax.text(0.0,-0.15,
        f'Naive free-node BO searches {R["DIM"]} dimensions ({R["D"]}D embedding \u00d7 {R["K"]} anchors). With ~50 noisy '
        'evaluations it never escapes the low-influence region;\nboth BO and random plateau near 90, far below top-degree. '
        'The constrained design (12-D, hub-restricted) reaches 183.',
        transform=ax.transAxes, fontsize=8.5, color=MUTE, va='top')
plt.tight_layout(); plt.savefig(f'{OUT}/bo_freenode_failure.png',bbox_inches='tight'); plt.close()
print('fig2 done')

# ---- FIG 3: diversity finally pays — but only when constrained ----
fig,ax=plt.subplots(figsize=(11,6))
variants=['Top-degree','Rule-tuning BO','Constrained\nfree-node BO']
spread=[175.2,179.0,183.1]
diversity=[1-0.20, 1-0.18, 1-0.135]   # 1 - overlap (proxy: more shared=less diverse). use seed-share as diversity inverse
share_div=[2/20, 2/20, 6/20]          # fraction of seeds that differ from top-degree
ax2=ax.twinx()
bars=ax.bar(np.arange(3)-0.0, spread, 0.5, color=[GREEN,BLUE,TEAL], alpha=0.85, edgecolor='white')
for b,v in zip(bars,spread): ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.0f}',ha='center',fontsize=10,fontweight='bold')
ax.plot(np.arange(3), [s*200 for s in share_div], color=RED, marker='D', ms=9, lw=2, label='Seed diversity (vs top-degree)')
for i,sd in enumerate(share_div):
    ax.text(i, sd*200+6, f'{sd:.0%}', ha='center', color=RED, fontsize=9)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(variants)
ax.set_ylabel('Influence spread (bars)')
ax.set_ylim(0,205)
ax.set_title('Diversity pays only inside the right search space', loc='left')
ax.spines['top'].set_visible(False)
ax.legend(loc='upper left', frameon=True, fontsize=9)
ax.text(0.0,-0.13,
        'Red diamonds: share of BO\u2019s seeds that differ from top-degree (a diversity proxy). The constrained free-node design '
        'achieves both the\nhighest diversity (30% of seeds replaced) AND the highest spread (183) \u2014 the synthesis the first two '
        'experiments were missing.',
        transform=ax.transAxes, fontsize=8.5, color=MUTE, va='top')
plt.tight_layout(); plt.savefig(f'{OUT}/bo_freenode_synthesis.png',bbox_inches='tight'); plt.close()
print('fig3 done')
