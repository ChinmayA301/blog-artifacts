import matplotlib.pyplot as plt, numpy as np, pickle
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi':150,'font.size':11,'axes.titlesize':14,
                     'axes.titleweight':'bold','axes.labelsize':11,'savefig.dpi':150})
BLUE='#4C72B0'; ORANGE='#DD8452'; GREEN='#55A868'; RED='#C44E52'; PURPLE='#8172B3'; MUTE='#9AA0A6'
OUT='figures'
os.makedirs(OUT, exist_ok=True)
R=pickle.load(open('/home/claude/bo_strong_results.pkl','rb'))
bo,rs=R['bo_curves'],R['rand_curves']

# values from the verified significance test
TD_M,TD_S=174.6,1.4; BO_M,BO_S=179.0,1.5

# FIG 1 convergence
fig,ax=plt.subplots(figsize=(11,6))
x=np.arange(1,bo.shape[1]+1)
for c,col,lab in [(bo,BLUE,'Bayesian Optimization'),(rs,ORANGE,'Random search')]:
    ax.plot(x,c.mean(0),color=col,lw=2.4,marker='o',ms=3,label=lab)
    ax.fill_between(x,c.min(0),c.max(0),color=col,alpha=0.16)
ax.axhline(R['b_topdeg'],color=GREEN,ls='--',lw=1.6,label='Top-degree heuristic')
ax.axhline(R['b_celf'],color=PURPLE,ls='-.',lw=1.6,label='CELF greedy')
ax.axhline(R['b_rand'],color=MUTE,ls=':',lw=1.6,label='Random seeds')
ax.set_xlabel('Objective evaluations (IC simulations)')
ax.set_ylabel('Best expected spread found (nodes activated)')
ax.set_title('Stronger design space: BO converges above top-degree and CELF',loc='left')
ax.legend(loc='center right',frameon=True,fontsize=9)
ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
ax.text(0.0,-0.15,f'Higgs retweet WCC ({R["graph_nodes"]:,} nodes / {R["graph_edges"]:,} edges). k={R["K"]}, IC p={R["IC_P"]}. '
        '4-knob design space (degree/k-core exponents, redundancy penalty, hop). Bands = min\u2013max over 5 runs.',
        transform=ax.transAxes,fontsize=8.5,color=MUTE,va='top')
plt.tight_layout();plt.savefig(f'{OUT}/bo_strong_convergence.png',bbox_inches='tight');plt.close()

# FIG 2 comparison with error bars + significance
fig,ax=plt.subplots(figsize=(11,6))
labels=['Random\nseeds','CELF\ngreedy','Top-degree','BO-optimized']
means=[R['b_rand'],R['b_celf'],TD_M,BO_M]
errs=[R['b_rand_std'],0,TD_S,BO_S]
cols=[MUTE,PURPLE,GREEN,BLUE]
yp=np.arange(len(labels))
bars=ax.barh(yp,means,xerr=errs,color=cols,edgecolor='white',alpha=0.92,
             error_kw=dict(ecolor='#444',capsize=4,lw=1.2))
ax.set_yticks(yp);ax.set_yticklabels(labels)
for b,m in zip(bars,means):
    ax.text(b.get_width()+2,b.get_y()+b.get_height()/2,f'{m:.0f}',va='center',fontsize=10)
# significance bracket between top-degree and BO
ax.annotate('',xy=(BO_M,3),xytext=(TD_M,2),arrowprops=dict(arrowstyle='-',color='#444',lw=1))
ax.text(BO_M+10,2.5,'+2.5%\np<0.001',fontsize=8.5,va='center',color='#444')
ax.set_xlabel('Expected influence spread (nodes activated, IC model)')
ax.set_title('BO beats top-degree by a small but statistically real margin',loc='left')
ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
ax.set_xlim(0,max(means)*1.22)
ax.text(0.0,-0.15,'BO\u2019s edge is significant (t=11.7, p<0.0001 over 30 high-precision estimates) but small. CELF underperforms here '
        'because\nlazy-greedy marginal gains are noisy under stochastic IC at this scale \u2014 a known practical failure mode.',
        transform=ax.transAxes,fontsize=8.5,color=MUTE,va='top')
plt.tight_layout();plt.savefig(f'{OUT}/bo_strong_comparison.png',bbox_inches='tight');plt.close()

# FIG 3 the honest mechanism: BO shares 18/20 seeds with top-degree
fig,ax=plt.subplots(figsize=(11,6))
cats=['Shared with\ntop-degree','BO-unique\nswaps']
vals=[18,2]
bars=ax.bar(cats,vals,color=[GREEN,BLUE],edgecolor='white',width=0.5,alpha=0.9)
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2,v+0.3,str(v),ha='center',fontsize=12,fontweight='bold')
ax.set_ylabel('Seeds (of k=20)')
ax.set_ylim(0,22)
ax.set_title('The mechanism, honestly: refinement, not diversity',loc='left')
ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
ax.text(0.0,-0.13,'The diversity hypothesis did not hold: BO\u2019s seed set overlaps top-degree by 18/20, and seed-region overlap is '
        f'identical ({R["ov_topdeg"]:.2f} vs {R["ov_bo"]:.2f}).\nBO\u2019s gain comes from swapping 2 weak hubs for better-placed nodes via '
        'exponent tuning \u2014 a real but modest effect. Higher IC p or a free node-level design space remain the next test.',
        transform=ax.transAxes,fontsize=8.5,color=MUTE,va='top')
plt.tight_layout();plt.savefig(f'{OUT}/bo_strong_mechanism.png',bbox_inches='tight');plt.close()
print('done')
