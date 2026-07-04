"""
Engineering Virality with Bayesian Optimization - stronger variant.

The weak version let BO only reweight centrality rankings, so its optimum
collapsed onto top-degree. Here BO searches a design space where overlap
between seeds' influence regions is the binding constraint, which is where
naive top-degree wastes budget on redundant hubs.

Design space (4 continuous knobs BO optimizes):
  a  - exponent on out-degree score          (how hub-greedy to be)
  b  - exponent on k-core score              (peripheral vs core)
  lam- redundancy penalty strength           (diversity pressure)
  hop- neighborhood radius for redundancy    (how far overlap is discounted)

Seed builder: greedy forward selection under a blended score that DISCOUNTS
candidates near already-chosen seeds (lam, hop). lam=0 reproduces pure
top-degree; lam>0 forces spatial spread. BO has to find the spread that
maximizes real IC spread.

Baselines: top-degree, random, and CELF (the standard near-optimal greedy
for influence maximization under IC). Beating/ matching CELF is the real bar.
"""
import gzip, pickle, time
import numpy as np
import networkx as nx
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

rng = np.random.default_rng(11)
H = pickle.load(open('rt_graph.pkl','rb'))
nodes = np.array(H.nodes())
N = len(nodes)
K = 20
IC_P = 0.12          # higher -> larger cascades -> overlap matters
MC = 80

deg = dict(H.out_degree())
deg_arr = np.array([deg[n] for n in nodes], dtype=float)
core = nx.core_number(H.to_undirected())
core_arr = np.array([core[n] for n in nodes], dtype=float)
succ = {n:list(H.successors(n)) for n in nodes}
idx = {n:i for i,n in enumerate(nodes)}

# undirected hop neighborhoods (for redundancy discount) - precompute 1-hop sets
und = H.to_undirected()
neigh1 = {n:set(und.neighbors(n)) for n in nodes}

def ic_spread(seeds, n_sims=MC):
    tot=0
    for _ in range(n_sims):
        active=set(seeds); frontier=list(seeds)
        while frontier:
            nxt=[]
            for u in frontier:
                for v in succ[u]:
                    if v not in active and rng.random()<IC_P:
                        active.add(v); nxt.append(v)
            frontier=nxt
        tot+=len(active)
    return tot/n_sims

# candidate pool: restrict to top 3000 by degree for tractable greedy selection
pool_idx = np.argpartition(-deg_arr, 3000)[:3000]
pool = nodes[pool_idx]
pdeg = deg_arr[pool_idx]; pcore = core_arr[pool_idx]
pdeg_z = (pdeg-pdeg.mean())/pdeg.std()
pcore_z = (pcore-pcore.mean())/pcore.std()

def build_seeds(a, b, lam, hop):
    """Greedy forward selection with redundancy discounting."""
    base_score = a*pdeg_z + b*pcore_z
    chosen=[]; chosen_neigh=set()
    avail = np.ones(len(pool), dtype=bool)
    for _ in range(K):
        score = base_score.copy()
        if lam>0 and chosen:
            # discount candidates whose neighborhood overlaps chosen set
            for j,n in enumerate(pool):
                if not avail[j]: continue
                if n in chosen_neigh:
                    score[j] -= lam
        score[~avail] = -1e9
        pick = int(np.argmax(score))
        chosen.append(pool[pick]); avail[pick]=False
        # expand redundancy region by hop
        nb = neigh1[pool[pick]]
        chosen_neigh |= nb
        if hop>=2:
            for m in list(nb):
                chosen_neigh |= neigh1.get(m,set())
    return chosen

def objective(theta):
    a,b,lam,hop = theta
    hop = 1 if hop<0.5 else 2
    return ic_spread(build_seeds(a,b,lam,hop))

bounds = np.array([[0.2,2.0],[0.0,1.5],[0.0,6.0],[0.0,1.0]])
def sample(r,n): return r.uniform(bounds[:,0], bounds[:,1], size=(n,4))

# ----- baselines -----
print('baselines...')
topdeg_seeds = [pool[i] for i in np.argsort(-pdeg)[:K]]
b_topdeg = ic_spread(topdeg_seeds, n_sims=400)
rand_vals=[ic_spread(list(rng.choice(nodes,K,replace=False)),n_sims=120) for _ in range(20)]
b_rand=np.mean(rand_vals); b_rand_std=np.std(rand_vals)

# CELF: standard lazy-greedy influence maximization
def celf(n_sims=80):
    import heapq
    # initial marginal gains
    gains=[]
    for n in pool:
        g = ic_spread([n], n_sims=20)
        heapq.heappush(gains, (-g, n, 0))
    S=[]; spread=0; last_spread=0
    while len(S)<K:
        neg_g, u, it = heapq.heappop(gains)
        if it==len(S):
            S.append(u)
            last_spread = ic_spread(S, n_sims=n_sims)
            continue
        else:
            mg = ic_spread(S+[u], n_sims=40) - last_spread
            heapq.heappush(gains,( -mg, u, len(S)))
    return S
print('running CELF (near-optimal greedy)...')
t0=time.time()
celf_seeds = celf()
b_celf = ic_spread(celf_seeds, n_sims=400)
print('  CELF done in %.0fs -> %.1f' % (time.time()-t0, b_celf))
print('  top-degree=%.1f random=%.1f' % (b_topdeg, b_rand))

# ----- BO vs random search, multiple runs -----
N_ITERS=44; N_RUNS=5
def ei(Xs, gp, ybest, xi=0.01):
    mu,sd=gp.predict(Xs,return_std=True); sd=np.maximum(sd,1e-9)
    imp=mu-ybest-xi; z=imp/sd
    return imp*norm.cdf(z)+sd*norm.pdf(z)

def run_bo(seed):
    r=np.random.default_rng(seed)
    X=sample(r,6); y=np.array([objective(x) for x in X])
    curve=[y.max()]; best_x=X[np.argmax(y)]
    for _ in range(N_ITERS-6):
        k=ConstantKernel(1.0)*Matern(length_scale=[0.5,0.5,1.0,0.3],nu=2.5)+WhiteKernel(noise_level=5.0)
        gp=GaussianProcessRegressor(kernel=k,normalize_y=True,n_restarts_optimizer=2,random_state=seed)
        gp.fit(X,y)
        cand=sample(r,1000); e=ei(cand,gp,y.max())
        xn=cand[np.argmax(e)]; yn=objective(xn)
        X=np.vstack([X,xn]); y=np.append(y,yn); curve.append(y.max())
        if yn>=y.max(): best_x=xn
    return np.array(curve), X[np.argmax(y)], y.max()

def run_rand(seed):
    r=np.random.default_rng(seed+200); best=-1; curve=[]
    for _ in range(N_ITERS-5):
        v=objective(sample(r,1)[0]); best=max(best,v); curve.append(best)
    return np.array(curve)

print('running BO (%dx%d)...' % (N_RUNS,N_ITERS))
bo_curves=[]; bo_best=-1; bo_best_x=None
for s in range(N_RUNS):
    c,x,v=run_bo(s); bo_curves.append(c)
    if v>bo_best: bo_best,bo_best_x=v,x
    print('  BO run %d best=%.1f theta=%s' % (s,v,np.round(x,2)))
rand_curves=[run_rand(s) for s in range(N_RUNS)]
L=min(min(len(c) for c in bo_curves),min(len(c) for c in rand_curves))
bo_curves=np.array([c[:L] for c in bo_curves]); rand_curves=np.array([c[:L] for c in rand_curves])

bo_seeds=build_seeds(bo_best_x[0],bo_best_x[1],bo_best_x[2],1 if bo_best_x[3]<0.5 else 2)
bo_final=ic_spread(bo_seeds,n_sims=400)

# measure seed-set diversity (mean pairwise hop / overlap proxy)
def overlap_frac(seeds):
    cover=set()
    for s in seeds: cover|=neigh1[s]
    union=sum(len(neigh1[s]) for s in seeds)
    return 1 - (len(cover)/union if union else 0)   # higher = more redundant
print('\nFINAL: top-degree=%.1f CELF=%.1f BO=%.1f random=%.1f'%(b_topdeg,b_celf,bo_final,b_rand))
print('overlap top-degree=%.2f BO=%.2f'%(overlap_frac(topdeg_seeds),overlap_frac(bo_seeds)))
print('BO theta=',np.round(bo_best_x,3))

pickle.dump(dict(bo_curves=bo_curves,rand_curves=rand_curves,
    b_topdeg=b_topdeg,b_celf=b_celf,b_rand=b_rand,b_rand_std=b_rand_std,
    bo_final=bo_final,bo_theta=bo_best_x,
    ov_topdeg=overlap_frac(topdeg_seeds),ov_bo=overlap_frac(bo_seeds),
    K=K,IC_P=IC_P,graph_nodes=H.number_of_nodes(),graph_edges=H.number_of_edges()),
    open('bo_strong_results.pkl','wb'))
print('saved.')
