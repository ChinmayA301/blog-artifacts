"""Fairer free-node BO: lower-dim search, degree-aware decoding, more budget.
Tests whether free node selection can be made competitive, or whether the
combinatorial penalty is fundamental at this evaluation budget."""
import pickle, time
import numpy as np, networkx as nx
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy import stats

rng=np.random.default_rng(31)
H=pickle.load(open('rt_graph.pkl','rb'))
nodes=np.array(H.nodes()); succ={n:list(H.successors(n)) for n in nodes}
deg=dict(H.out_degree()); deg_arr=np.array([deg[n] for n in nodes],float)
K,IC_P,MC=20,0.12,80

# pool: top 400 by degree -> BO picks diverse SUBSETS of strong nodes
pool_idx=np.argpartition(-deg_arr,400)[:400]; pool=nodes[pool_idx]
pdeg=deg_arr[pool_idx]
und=H.subgraph(pool).to_undirected()
neigh1={n:set(und.neighbors(n)) for n in pool}
pdeg_z=(pdeg-pdeg.mean())/pdeg.std()

def ic(seeds,n=MC):
    tot=0
    for _ in range(n):
        a=set(seeds);fr=list(seeds)
        while fr:
            nn=[]
            for u in fr:
                for v in succ[u]:
                    if v not in a and rng.random()<IC_P: a.add(v);nn.append(v)
            fr=nn
        tot+=len(a)
    return tot/n

# BO searches a per-node logit bias over the 400-pool, compressed via a
# random projection to keep dims tractable, plus a global diversity weight.
# decode: greedy pick by (degree + bias - lam*overlap)
P=12  # latent dims
R=rng.normal(size=(P,len(pool)))/np.sqrt(P)  # random projection latent->node
def decode(theta):
    z=theta[:P]; lam=theta[P]*4
    bias=z@R
    score0=pdeg_z+bias
    chosen=[];cn=set()
    av=np.ones(len(pool),bool)
    for _ in range(K):
        sc=score0.copy()
        if lam>0 and chosen:
            for j,nd in enumerate(pool):
                if av[j] and nd in cn: sc[j]-=lam
        sc[~av]=-1e9; p=int(np.argmax(sc)); chosen.append(pool[p]);av[p]=False
        cn|=neigh1[pool[p]]
    return chosen
DIM=P+1
def obj(t): return ic(decode(t))
def smp(r,n): return np.hstack([r.normal(size=(n,P)), r.uniform(0,1,size=(n,1))])

td=[pool[i] for i in np.argsort(-pdeg)[:K]]; b_td=ic(td,400)
def ei(Xs,gp,yb,xi=0.01):
    mu,sd=gp.predict(Xs,return_std=True);sd=np.maximum(sd,1e-9)
    im=mu-yb-xi;z=im/sd;return im*norm.cdf(z)+sd*norm.pdf(z)
def run(seed):
    r=np.random.default_rng(seed);X=smp(r,14);y=np.array([obj(x) for x in X]);c=[y.max()]
    for _ in range(70):
        k=ConstantKernel(1.0)*Matern(length_scale=1.0,nu=2.5)+WhiteKernel(noise_level=5.0)
        gp=GaussianProcessRegressor(kernel=k,normalize_y=True,n_restarts_optimizer=1,random_state=seed).fit(X,y)
        cand=smp(r,1500);e=ei(cand,gp,y.max());xn=cand[np.argmax(e)];yn=obj(xn)
        X=np.vstack([X,xn]);y=np.append(y,yn);c.append(y.max())
    return np.array(c),X[np.argmax(y)],y.max()
print('top-degree=%.1f, running fair free-node BO...'%b_td)
best=-1;bx=None;curves=[]
for s in range(3):
    c,x,v=run(s);curves.append(c)
    if v>best:best,bx=v,x
    print('  run %d best=%.1f'%(s,v))
bo_seeds=decode(bx);bo_f=ic(bo_seeds,400);shared=len(set(bo_seeds)&set(td))
td_s=np.array([ic(td,150) for _ in range(25)]);bo_s=np.array([ic(bo_seeds,150) for _ in range(25)])
t,p=stats.ttest_ind(bo_s,td_s)
print('FAIR free-node: td=%.1f BO=%.1f shared=%d/%d'%(b_td,bo_f,shared,K))
print('td %.1f±%.1f BO %.1f±%.1f diff=%.1f(%.1f%%) p=%.4f'%(
    td_s.mean(),td_s.std(),bo_s.mean(),bo_s.std(),bo_s.mean()-td_s.mean(),
    100*(bo_s.mean()-td_s.mean())/td_s.mean(),p))
