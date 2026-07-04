"""
#24 BO-meets-graphs — companion visuals to the virality post.
Real data: Higgs retweet graph + the constrained free-node BO setup.

Two figures:
  1. Acquisition-over-nodes: embed candidate nodes in 2D, show BO's posterior
     mean (predicted spread) as a surface and mark where BO chose to probe
     next vs where top-degree sits. Illustrates BO exploring the graph.
  2. Combinatorial blow-up: why you can't brute-force seed selection —
     search-space size vs k, with the regimes labeled.
"""
import pickle, numpy as np, networkx as nx
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import spectral_embedding
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi':150,'font.size':11,'axes.titlesize':14,
                     'axes.titleweight':'bold','axes.labelsize':11,'savefig.dpi':150})
BLUE='#4C72B0'; ORANGE='#DD8452'; GREEN='#55A868'; RED='#C44E52'; PURPLE='#8172B3'; MUTE='#9AA0A6'
OUT='figures'
os.makedirs(OUT, exist_ok=True)

H = pickle.load(open('rt_graph.pkl','rb'))
nodes = np.array(H.nodes())
deg = dict(H.out_degree()); deg_arr = np.array([deg[n] for n in nodes], float)

# candidate pool + 2D spectral embedding for visualization
pool_idx = np.argpartition(-deg_arr, 800)[:800]
pool = nodes[pool_idx]
sub = H.subgraph(pool).to_undirected()
cc = max(nx.connected_components(sub), key=len)
sub_cc = sub.subgraph(cc)
emb_nodes = np.array(sub_cc.nodes())
A = nx.to_scipy_sparse_array(sub_cc, nodelist=emb_nodes, format='csr', dtype=float)
A.indices = A.indices.astype(np.int32); A.indptr = A.indptr.astype(np.int32)
E = spectral_embedding(A, n_components=2, random_state=0)
E = (E - E.min(0)) / (np.ptp(E, axis=0) + 1e-9)
edeg = np.array([deg[n] for n in emb_nodes], float)
print('embedded', len(emb_nodes), 'nodes in 2D')

# ---- FIG 1: honest node scatter, sized/coloured by real out-degree ----
# Use a force-directed layout on the actual subgraph (spectral was degenerate
# in 2D). No interpolated field — every point is a real node.
import networkx as nx
np.random.seed(0)
# layout on the induced subgraph of the embedding nodes
Gsub = sub_cc
pos = nx.spring_layout(Gsub, k=0.15, iterations=40, seed=1)
P = np.array([pos[n] for n in emb_nodes])
P = (P - P.min(0)) / (np.ptp(P, axis=0) + 1e-9)

fig, ax = plt.subplots(figsize=(11,7.5))
# size and colour by out-degree (the real value signal)
sizes = 20 + 380*(edeg/edeg.max())
sc = ax.scatter(P[:,0], P[:,1], s=sizes, c=np.log1p(edeg), cmap='viridis',
                alpha=0.75, edgecolors='white', linewidths=0.3, zorder=2)

td_idx = np.argsort(-edeg)[:20]
ax.scatter(P[td_idx,0], P[td_idx,1], s=200, marker='o', facecolors='none',
           edgecolors=RED, linewidths=2.4, zorder=4, label='Top-degree seeds (clustered)')

chosen=[]; order=np.argsort(-edeg)
for i in order:
    if len(chosen)>=20: break
    if all(np.hypot(*(P[i]-P[j]))>0.14 for j in chosen):
        chosen.append(i)
chosen=np.array(chosen)
ax.scatter(P[chosen,0], P[chosen,1], s=150, marker='D', facecolors='none',
           edgecolors='#111', linewidths=2.0, zorder=5, label='Diverse-subset seeds (spread)')

cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
cbar.set_label('log out-degree (influence potential)', fontsize=9)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('Where the strategies seed: hub cluster vs. spread across the graph', loc='left')
ax.legend(loc='upper right', frameon=True, fontsize=9)
ax.text(0,-0.06,
        f'Higgs retweet graph, {len(emb_nodes)}-node high-degree subgraph, force-directed layout. Point size and colour = real '
        'out-degree.\nTop-degree seeds (red rings) concentrate in the densest hub region; the diverse-subset strategy (black '
        'diamonds) keeps high degree while spreading across the graph \u2014 the behaviour that won in the main post.',
        transform=ax.transAxes, fontsize=8.5, color=MUTE, va='top')
plt.tight_layout(); plt.savefig(f'{OUT}/graphbo_acquisition_landscape.png', bbox_inches='tight'); plt.close()
print('fig1 done')

# ---- FIG 2: combinatorial blow-up ----
from math import comb
fig, ax = plt.subplots(figsize=(11,6))
Ns = [800, 223833]
ks = np.arange(1, 31)
for N, col, lab in [(800, ORANGE, 'candidate pool (N=800)'), (223833, BLUE, 'full graph (N=223,833)')]:
    sizes = [comb(N, int(k)) for k in ks]
    ax.semilogy(ks, sizes, lw=2.4, color=col, marker='o', ms=3, label=lab)
ax.axhline(1e6, color=MUTE, ls=':', lw=1.4)
ax.text(1, 1.5e6, 'a million evaluations', fontsize=8.5, color=MUTE)
ax.axvline(20, color=RED, ls='--', lw=1.6)
ax.text(20.3, 1e30, 'k = 20\n(our budget)', color=RED, fontsize=9)
ax.set_xlabel('Seed set size k'); ax.set_ylabel('Number of possible seed sets (log scale)')
ax.set_title('Why you cannot brute-force seed selection', loc='left')
ax.legend(loc='lower right', frameon=True, fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.text(0,-0.15,
        'At k=20 the full graph admits ~10^90 seed sets \u2014 more than atoms in the observable universe. Even the 800-node '
        'candidate pool gives ~10^41.\nThis is why BO searches a low-dimensional *parameterization* of seed sets, never the '
        'sets themselves.',
        transform=ax.transAxes, fontsize=8.5, color=MUTE, va='top')
plt.tight_layout(); plt.savefig(f'{OUT}/graphbo_combinatorial.png', bbox_inches='tight'); plt.close()
print('fig2 done | full-graph C(223833,20) ~ 1e%d' % int(np.log10(float(comb(223833,20)))))
