"""Shared chart style + pickle loader. Every number on every figure comes from
the pipeline pickles - nothing is hardcoded."""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'font.size': 11, 'axes.titlesize': 14,
                     'axes.titleweight': 'bold', 'axes.labelsize': 11, 'savefig.dpi': 150})
BLUE = '#4C72B0'; ORANGE = '#DD8452'; GREEN = '#55A868'; RED = '#C44E52'
PURPLE = '#8172B3'; MUTE = '#9AA0A6'; TEAL = '#17A2B8'

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, '..', 'pipeline')
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

STAGE_FILES = {1: 'bo_results.pkl', 2: 'bo_strong_results.pkl',
               3: 'bo_freenode_results.pkl', 4: 'bo_constrained_results.pkl'}


def load(stage):
    path = os.path.join(PIPE, STAGE_FILES[stage])
    if not os.path.exists(path):
        raise SystemExit('%s missing - run the pipeline first' % path)
    R = pickle.load(open(path, 'rb'))
    if R.get('QUICK'):
        print('WARNING: %s came from a QUICK smoke-test run' % STAGE_FILES[stage])
    return R


def clean(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


def caption(ax, text, y=-0.15):
    ax.text(0.0, y, text, transform=ax.transAxes, fontsize=8.5, color=MUTE, va='top')


def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(OUT, name), bbox_inches='tight'); plt.close()
    print('wrote', name)


def convergence(ax, R, bo_label='Bayesian Optimization'):
    """BO vs random search on a shared x-axis: point i = best after i
    objective evaluations, for BOTH methods (initial design included)."""
    bo, rs = R['bo_curves'], R['rs_curves']
    assert bo.shape[1] == rs.shape[1], 'curves must share an evaluation axis'
    x = np.arange(1, bo.shape[1] + 1)
    for c, col, lab, mk in [(bo, BLUE, bo_label, 'o'), (rs, ORANGE, 'Random search (same budget)', 's')]:
        ax.plot(x, c.mean(0), color=col, lw=2.4, marker=mk, ms=3, label=lab)
        ax.fill_between(x, c.min(0), c.max(0), color=col, alpha=0.16)
    B = R['baselines']
    ax.axhline(B['td_final'], color=GREEN, ls='--', lw=1.6, label='Top-degree (%.0f)' % B['td_final'])
    ax.set_xlabel('Objective evaluations (each = %d IC simulations)' % R['MC'])
    ax.set_ylabel('Best noisy spread found so far')


def setup_line(R):
    return ('Higgs retweet WCC (%s nodes / %s edges), edges oriented retweeted → retweeter. k=%d, IC p=%s. '
            'Bands = min–max over %d runs.' % (f'{R["graph_nodes"]:,}', f'{R["graph_edges"]:,}',
                                                   R['K'], R['IC_P'], R['bo_curves'].shape[0]))


def pct(a, b):
    return 100.0 * (a - b) / b
