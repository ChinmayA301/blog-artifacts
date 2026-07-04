"""
Free node-level BO for influence maximization.

Prior variants let BO tune a centrality RULE -> optimum collapsed near top-degree.
Here BO selects actual nodes, so it can exploit influence-region diversity that
no degree-ranking rule can express.

Search-space construction:
  1. Embed candidate nodes into d-dim space via spectral embedding of the
     retweet graph (Laplacian eigenmaps). Nodes close in this space have
     correlated influence regions; nodes far apart cover different parts of
     the graph. This is the geometry diversity needs.
  2. BO searches K "anchor points" in the d-dim space (K*d continuous dims).
     Each anchor decodes to its nearest candidate node -> a concrete seed set.
  3. Objective = IC spread of that seed set (real simulation).

Because anchors can be placed anywhere, BO can build a SPREAD-OUT seed set
(low influence overlap) or a CLUSTERED one (high degree, high overlap) and
let the data decide. Baselines: top-degree, CELF, random, and the
rule-tuning BO winner from the prior run (179.0).
"""
import pickle, time
import numpy as np
import networkx as nx
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

rng = np.random.default_rng(23)
H = pickle.load(open('rt_graph.pkl','rb'))
nodes = np.array(H.nodes())
succ = {n:list(H.successors(n)) for n in nodes}
deg = dict(H.out_degree())
deg_arr = np.array([deg[n] for n in nodes], float)
K, IC_P, MC = 20, 0.12, 80

# ---- candidate pool: top 1500 by degree (where useful seeds live) ----
pool_idx = np.argpartition(-deg_arr, 1500)[:1500]
pool = nodes[pool_idx]
pdeg = deg_arr[pool_idx]
pos = {n:i for i,n in enumerate(pool)}
print('pool=%d nodes' % len(pool))

# ---- spectral embedding of the pool's induced subgraph (undirected) ----
t0=time.time()
sub = H.subgraph(pool).to_undirected()
# ensure connected enough for embedding: use largest CC of the subgraph
cc = max(nx.connected_components(sub), key=len)
sub_cc = sub.subgraph(cc)
emb_nodes = np.array(sub_cc.nodes())
A = nx.to_scipy_sparse_array(sub_cc, nodelist=emb_nodes, format='csr', dtype=float)
A.indices = A.indices.astype(np.int32); A.indptr = A.indptr.astype(np.int32)
D = 6
EMB = spectral_embedding(A, n_components=D, random_state=0)
# normalize embedding dims to [0,1] for BO bounds
EMB = (EMB - EMB.min(0)) / (np.ptp(EMB, axis=0) + 1e-9)
nn = NearestNeighbors(n_neighbors=1).fit(EMB)
print('embedding: %d nodes in %dD (%.1fs)' % (len(emb_nodes), D, time.time()-t0))

emb_deg = np.array([deg[n] for n in emb_nodes], float)

def ic_spread(seeds, n_sims=MC):
    tot=0
    for _ in range(n_sims):
        active=set(seeds); fr=list(seeds)
        while fr:
            nx_=[]
            for u in fr:
                for v in succ[u]:
                    if v not in active and rng.random()<IC_P:
                        active.add(v); nx_.append(v)
            fr=nx_
        tot+=len(active)
    return tot/n_sims

def decode(anchor_flat):
    """K*D vector -> K nearest distinct nodes."""
    anchors = anchor_flat.reshape(K, D)
    _, idxs = nn.kneighbors(anchors, n_neighbors=8)
    seeds=[]; used=set()
    for row in idxs:
        for j in row:
            if j not in used:
                used.add(j); seeds.append(emb_nodes[j]); break
    # pad if collisions reduced count
    while len(seeds)<K:
        j=rng.integers(len(emb_nodes))
        if j not in used: used.add(j); seeds.append(emb_nodes[j])
    return seeds[:K]

def objective(anchor_flat):
    return ic_spread(decode(anchor_flat))

DIM = K*D
def sample(r,n): return r.uniform(0,1,size=(n,DIM))

# ---- overlap metric ----
und = H.to_undirected(); neigh1={n:set(und.neighbors(n)) for n in emb_nodes}
def overlap_frac(seeds):
    cover=set(); tot=0
    for s in seeds:
        nb=neigh1.get(s,set()); cover|=nb; tot+=len(nb)
    return 1-(len(cover)/tot if tot else 0)

# ---- baselines (on embedding node set, for fair comparison) ----
print('baselines...')
td=[emb_nodes[i] for i in np.argsort(-emb_deg)[:K]]
b_td=ic_spread(td,400)
randv=[ic_spread(list(rng.choice(emb_nodes,K,replace=False)),120) for _ in range(20)]
b_rand=np.mean(randv); b_rand_std=np.std(randv)
print('  top-degree=%.1f random=%.1f' % (b_td,b_rand))

# ---- BO vs random search ----
N_ITERS=60; N_RUNS=4
def ei(Xs,gp,yb,xi=0.01):
    mu,sd=gp.predict(Xs,return_std=True); sd=np.maximum(sd,1e-9)
    imp=mu-yb-xi; z=imp/sd
    return imp*norm.cdf(z)+sd*norm.pdf(z)

def run_bo(seed):
    r=np.random.default_rng(seed)
    X=sample(r,12); y=np.array([objective(x) for x in X])
    curve=[y.max()]
    for _ in range(N_ITERS-12):
        k=ConstantKernel(1.0)*Matern(length_scale=1.0,nu=2.5)+WhiteKernel(noise_level=5.0)
        gp=GaussianProcessRegressor(kernel=k,normalize_y=True,n_restarts_optimizer=1,random_state=seed)
        gp.fit(X,y)
        cand=sample(r,1500); e=ei(cand,gp,y.max())
        xn=cand[np.argmax(e)]; yn=objective(xn)
        X=np.vstack([X,xn]); y=np.append(y,yn); curve.append(y.max())
    return np.array(curve), X[np.argmax(y)], y.max()

def run_rand(seed):
    r=np.random.default_rng(seed+300); best=-1; c=[]
    for _ in range(N_ITERS-11):
        v=objective(sample(r,1)[0]); best=max(best,v); c.append(best)
    return np.array(c)

print('running free-node BO (%dx%d, %dD search)...' % (N_RUNS,N_ITERS,DIM))
bo_curves=[]; bo_best=-1; bo_x=None
for s in range(N_RUNS):
    c,x,v=run_bo(s); bo_curves.append(c)
    if v>bo_best: bo_best,bo_x=v,x
    print('  run %d best=%.1f' % (s,v))
rand_curves=[run_rand(s) for s in range(N_RUNS)]
L=min(min(len(c) for c in bo_curves),min(len(c) for c in rand_curves))
bo_curves=np.array([c[:L] for c in bo_curves]); rand_curves=np.array([c[:L] for c in rand_curves])

bo_seeds=decode(bo_x)
bo_final=ic_spread(bo_seeds,400)
shared=len(set(bo_seeds)&set(td))
print('\nFINAL free-node: top-degree=%.1f BO=%.1f random=%.1f'%(b_td,bo_final,b_rand))
print('overlap: top-degree=%.2f BO=%.2f | shared seeds=%d/%d'%(overlap_frac(td),overlap_frac(bo_seeds),shared,K))

# significance: 30 high-precision estimates each
from scipy import stats
td_s=np.array([ic_spread(td,150) for _ in range(30)])
bo_s=np.array([ic_spread(bo_seeds,150) for _ in range(30)])
t,p=stats.ttest_ind(bo_s,td_s)
print('top-degree %.1f±%.1f  BO %.1f±%.1f  diff=%.1f(%.1f%%) t=%.2f p=%.4f'%(
    td_s.mean(),td_s.std(),bo_s.mean(),bo_s.std(),
    bo_s.mean()-td_s.mean(),100*(bo_s.mean()-td_s.mean())/td_s.mean(),t,p))

pickle.dump(dict(bo_curves=bo_curves,rand_curves=rand_curves,
    b_td=b_td,b_rand=b_rand,b_rand_std=b_rand_std,bo_final=bo_final,
    td_mean=td_s.mean(),td_std=td_s.std(),bo_mean=bo_s.mean(),bo_std=bo_s.std(),
    tstat=t,pval=p,shared=shared,
    ov_td=overlap_frac(td),ov_bo=overlap_frac(bo_seeds),
    K=K,IC_P=IC_P,DIM=DIM,D=D,pool=len(emb_nodes),
    prior_ruleBO=179.0,b_celf=149.6,
    graph_nodes=H.number_of_nodes(),graph_edges=H.number_of_edges()),
    open('bo_freenode_results.pkl','wb'))
print('saved.')
