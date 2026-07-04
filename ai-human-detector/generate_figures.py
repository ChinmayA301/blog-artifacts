"""
#3 AI vs human text detector.
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
# #3 — AI vs human imperfection detector
# Synthetic feature model: human text has higher burstiness + higher edit
# entropy + higher perplexity variance; AI text is smoother. We build a
# separable-but-overlapping 3-feature dataset, fit logistic regression,
# and show ROC + calibration + where it fails.
# =====================================================================
n = 1200
# human (label 1): higher, more variable imperfection features
human = np.column_stack([
    rng.normal(0.62,0.14,n),   # burstiness
    rng.normal(0.55,0.16,n),   # edit entropy
    rng.normal(0.58,0.18,n),   # perplexity variance
])
ai = np.column_stack([
    rng.normal(0.44,0.12,n),
    rng.normal(0.40,0.13,n),
    rng.normal(0.38,0.14,n),
])
X = np.vstack([human,ai]); y = np.hstack([np.ones(n),np.zeros(n)])
idx = rng.permutation(len(y)); X,y = X[idx],y[idx]
split=int(0.7*len(y))
Xtr,Xte,ytr,yte = X[:split],X[split:],y[:split],y[split:]
clf = LogisticRegression().fit(Xtr,ytr)
p = clf.predict_proba(Xte)[:,1]
fpr,tpr,_ = roc_curve(yte,p); AUC=auc(fpr,tpr)

fig, axes = plt.subplots(1,3, figsize=(15,5))
# feature distribution
ax=axes[0]
ax.hist(human[:,0],bins=30,alpha=0.6,color=GREEN,label='Human',density=True)
ax.hist(ai[:,0],bins=30,alpha=0.6,color=PURPLE,label='AI',density=True)
ax.set_xlabel('Burstiness feature'); ax.set_ylabel('Density')
ax.set_title('1. Feature separation',loc='left',fontsize=12)
ax.legend(frameon=True,fontsize=9); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
# ROC
ax=axes[1]
ax.plot(fpr,tpr,color=BLUE,lw=2.4,label=f'AUC = {AUC:.3f}')
ax.plot([0,1],[0,1],color=MUTE,ls='--',lw=1.2)
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title('2. ROC',loc='left',fontsize=12)
ax.legend(loc='lower right',frameon=True,fontsize=9); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
# calibration
ax=axes[2]
frac_pos,mean_pred = calibration_curve(yte,p,n_bins=10)
ax.plot(mean_pred,frac_pos,color=ORANGE,lw=2.2,marker='o',ms=5,label='Detector')
ax.plot([0,1],[0,1],color=MUTE,ls='--',lw=1.2,label='Perfect calibration')
ax.set_xlabel('Predicted P(human)'); ax.set_ylabel('Observed fraction human')
ax.set_title('3. Calibration',loc='left',fontsize=12)
ax.legend(loc='upper left',frameon=True,fontsize=9); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('An AI-text detector, evaluated honestly  (synthetic data)',x=0.09,ha='left',fontsize=14,fontweight='bold')
fig.text(0.09,-0.02,
    f'Logistic detector on three "imperfection" features (burstiness, edit entropy, perplexity variance). AUC={AUC:.3f} looks '
    'strong, but the overlap in panel 1 is\nwhere real misclassifications live: heavily-edited AI text and terse human text sit '
    'in the confusion zone. Synthetic data; the evaluation pipeline is the artifact.',
    fontsize=8.5,color=MUTE,va='top')
plt.tight_layout(rect=[0,0.02,1,0.96]); plt.savefig(f'{OUT}/detector_evaluation.png',bbox_inches='tight'); plt.close()
print('detector done | AUC=%.3f'%AUC)

