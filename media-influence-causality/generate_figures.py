"""
#8 media influence cross-correlation & DAG.
Both use SYNTHETIC-but-labeled data with the REAL analysis method. Captions
state this explicitly.
"""
import os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi':150,'font.size':11,'axes.titlesize':14,
                     'axes.titleweight':'bold','axes.labelsize':11,'savefig.dpi':150})
BLUE='#4C72B0'; ORANGE='#DD8452'; GREEN='#55A868'; RED='#C44E52'; PURPLE='#8172B3'; MUTE='#9AA0A6'
OUT='figures'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(202)

# =====================================================================
# #8 — media influence: cross-correlation with lead/lag + causal DAG
# Synthetic: a "media export volume" series drives an "influence index"
# with a lag, plus noise and a confounder. Show the CCF (where the peak
# lag sits) and a DAG that keeps the causal claim honest.
# =====================================================================
T=180
media = np.cumsum(rng.normal(0,1,T)) + 5*np.sin(np.arange(T)/12)
lag=6
# influence at time t depends on media at time t-lag => media LEADS by +lag
influence = np.empty(T)
conf = 0.3*np.cumsum(rng.normal(0,1,T))
for t in range(T):
    src = media[t-lag] if t-lag>=0 else 0.0
    influence[t] = 0.7*src + rng.normal(0,2) + conf[t]
def ccf(lead,follow,maxlag=24):
    a=(lead-lead.mean())/lead.std(); b=(follow-follow.mean())/follow.std()
    lags=range(-maxlag,maxlag+1); out=[]
    for L in lags:
        # positive L: does lead[t] predict follow[t+L]? (lead leads by L)
        if L>0: out.append(np.corrcoef(a[:-L],b[L:])[0,1])
        elif L<0: out.append(np.corrcoef(a[-L:],b[:L])[0,1])
        else: out.append(np.corrcoef(a,b)[0,1])
    return np.array(list(lags)),np.array(out)
lags,corr = ccf(media,influence)
peak=lags[np.argmax(corr)]

fig, (axL,axR) = plt.subplots(1,2, figsize=(14,5.5), gridspec_kw={'width_ratios':[1.1,1]})
# CCF
axL.stem(lags,corr,linefmt=BLUE,markerfmt='o',basefmt=' ')
axL.axvline(peak,color=RED,ls='--',lw=1.6)
axL.annotate(f'peak at lag +{peak}\n(media leads)', xy=(peak,corr.max()),
             xytext=(peak+6,corr.max()*0.78), color=RED, fontsize=9,
             arrowprops=dict(arrowstyle='->',color=RED,lw=1))
ci=1.96/np.sqrt(T)
axL.axhspan(-ci,ci,color=MUTE,alpha=0.15)
axL.set_xlabel('Lag (days) — positive = media export leads influence')
axL.set_ylabel('Cross-correlation')
axL.set_title('Media exports lead the influence index  (synthetic data)',loc='left',fontsize=12.5)
axL.spines['top'].set_visible(False); axL.spines['right'].set_visible(False)
axL.text(0,-0.16,'Shaded band = 95% no-correlation interval. A clear positive peak at a positive lag is *consistent with* media\n'
    'exports leading influence \u2014 but correlation at a lag is not proof of causation, which is what the diagram addresses.',
    transform=axL.transAxes,fontsize=8.5,color=MUTE,va='top')

# DAG
axR.axis('off'); axR.set_xlim(0,10); axR.set_ylim(0,10)
def node(x,y,txt,c=BLUE):
    axR.add_patch(Rectangle((x-1.3,y-0.6),2.6,1.2,fc='white',ec=c,lw=2,zorder=3))
    axR.text(x,y,txt,ha='center',va='center',fontsize=9.5,zorder=4)
def arrow(x1,y1,x2,y2,c='#333',style='-'):
    axR.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=16,
                  color=c,lw=1.8,ls=style,zorder=2,shrinkA=18,shrinkB=18))
node(2.5,8,'Media\nexports',BLUE)
node(7.5,8,'Global\ninfluence',GREEN)
node(5,3,'Economic /\nsoft power\n(confounder)',ORANGE)
arrow(2.5,8,7.5,8,'#333')
axR.text(5,8.5,'hypothesized causal path',ha='center',fontsize=8.5,color='#333')
arrow(5,3,2.5,8,ORANGE,'--')
arrow(5,3,7.5,8,ORANGE,'--')
axR.text(3.0,5.3,'confounds',fontsize=8,color=ORANGE,rotation=64)
arrow(7.5,7.4,2.7,7.4,RED,':')
axR.text(5,6.9,'reverse causation?',ha='center',fontsize=8.5,color=RED)
axR.set_title('The identification problem',loc='left',fontsize=12.5)
fig.text(0.55,-0.02,'A lagged correlation is compatible with three stories: media \u2192 influence, a shared confounder driving both,\n'
    'or influence \u2192 media. Separating them needs an instrument or natural experiment, not a bigger correlation.',
    fontsize=8.5,color=MUTE,va='top')
plt.tight_layout(); plt.savefig(f'{OUT}/media_influence_ccf_dag.png',bbox_inches='tight'); plt.close()
print('media done | peak lag=%d'%peak)
