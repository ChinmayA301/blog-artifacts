"""
#11 SignalGraph — GitHub fake-star detection.

NOTE ON DATA: GitHub's per-stargazer timestamp API requires authentication and
was rate-limited in this environment, so this runs on a SYNTHETIC star-history
dataset with planted inorganic bursts. The DETECTION METHOD is the real
contribution; the data is illustrative and labeled as such in every caption.

Method: model daily star counts, detect anomalous velocity via a robust
z-score on day-over-day gains (median/MAD, resistant to the spikes themselves),
flag contiguous anomalous days as "burst" events. Then show how credibility-
adjusting (discounting burst stars) reshuffles a leaderboard.
"""
import os, numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi':150,'font.size':11,'axes.titlesize':14,
                     'axes.titleweight':'bold','axes.labelsize':11,'savefig.dpi':150})
BLUE='#4C72B0'; ORANGE='#DD8452'; GREEN='#55A868'; RED='#C44E52'; PURPLE='#8172B3'; MUTE='#9AA0A6'
OUT='figures'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(101)

# ---- synthetic star histories: organic growth + planted bursts ----
days = 365
t = np.arange(days)
dates = [datetime(2025,7,1)+timedelta(days=int(i)) for i in t]

def organic(rate0, growth):
    daily = rate0*np.exp(growth*t/days) * rng.gamma(2.0, 0.5, size=days)
    return daily

def inject_burst(daily, start, length, size):
    d = daily.copy()
    d[start:start+length] += size/length * rng.uniform(0.7,1.3,size=length)
    return d

# repo A: organic. repo B: organic + two fake bursts.
A = organic(8, 1.2)
B = organic(6, 1.0)
B = inject_burst(B, 90, 4, 1800)   # sharp 4-day spike
B = inject_burst(B, 250, 3, 1400)  # second spike
A_cum, B_cum = np.cumsum(A), np.cumsum(B)

# ---- robust velocity-anomaly detector (median/MAD z on daily gains) ----
def detect(daily, thresh=5.0, min_run=2):
    med = np.median(daily)
    mad = np.median(np.abs(daily-med)) + 1e-9
    z = 0.6745*(daily-med)/mad
    raw_flag = z > thresh
    # burst-clustering: only keep anomalies that are part of a run of >=min_run
    flagged = np.zeros_like(raw_flag)
    i = 0
    while i < len(raw_flag):
        if raw_flag[i]:
            j = i
            while j < len(raw_flag) and raw_flag[j]:
                j += 1
            if j-i >= min_run:
                flagged[i:j] = True
            i = j
        else:
            i += 1
    return z, flagged

zB, flagB = detect(B)
zA, flagA = detect(A)

# ---- FIG 1: star velocity with flagged bursts ----
fig, (ax1,ax2) = plt.subplots(2,1, figsize=(11,7.5), sharex=True,
                              gridspec_kw={'height_ratios':[2,1]})
ax1.plot(dates, B, color=BLUE, lw=1.3, label='Daily new stars')
burst_days = np.where(flagB)[0]
ax1.scatter([dates[i] for i in burst_days], B[burst_days], color=RED, s=40, zorder=5,
            label='Flagged as inorganic')
# shade burst windows
for s,l in [(90,4),(250,3)]:
    ax1.axvspan(dates[s], dates[s+l], color=RED, alpha=0.08)
ax1.set_ylabel('New stars / day')
ax1.set_title('Velocity anomaly detection on a repo\u2019s star history  (synthetic data)', loc='left')
ax1.legend(loc='upper right', frameon=True, fontsize=9)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

ax2.plot(dates, zB, color=PURPLE, lw=1.2)
ax2.axhline(5.0, color=RED, ls='--', lw=1.4, label='anomaly threshold (z=5)')
ax2.fill_between(dates, 5.0, zB, where=zB>5.0, color=RED, alpha=0.25)
ax2.set_ylabel('Robust z-score')
ax2.legend(loc='upper right', frameon=True, fontsize=8.5)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax2.text(0,-0.32,
        'Median/MAD z-score on day-over-day star gains \u2014 robust to the spikes themselves. Two planted 3\u20134 day bursts '
        '(shaded) exceed z=5 and are flagged;\norganic growth stays well below. Synthetic star history; the detector is the real artifact.',
        transform=ax2.transAxes, fontsize=8.5, color=MUTE, va='top')
plt.tight_layout(); plt.savefig(f'{OUT}/signalgraph_velocity.png', bbox_inches='tight'); plt.close()
print('fig1 done | B flagged days:', flagB.sum(), '| A flagged days:', flagA.sum())

# ---- FIG 2: credibility-adjusted ranking reshuffle ----
# a small field of repos, some with inorganic inflation; show raw vs adjusted rank
repos = ['repo-alpha','repo-beta','repo-gamma','repo-delta','repo-epsilon','repo-zeta']
raw   = np.array([9200, 8600, 8100, 7400, 6900, 6200])
# fraction of stars attributable to flagged bursts (0 = clean)
burst_frac = np.array([0.02, 0.45, 0.05, 0.38, 0.03, 0.08])
adjusted = raw*(1-burst_frac)
raw_rank = raw.argsort()[::-1].argsort()+1
adj_rank = adjusted.argsort()[::-1].argsort()+1

fig, ax = plt.subplots(figsize=(11,6.5))
for i,r in enumerate(repos):
    col = RED if burst_frac[i]>0.2 else GREEN
    ax.plot([0,1],[raw_rank[i],adj_rank[i]], color=col, lw=2, alpha=0.8, marker='o', ms=8)
    ax.text(-0.03, raw_rank[i], r, ha='right', va='center', fontsize=9.5)
    ax.text(1.03, adj_rank[i], f'{r}  ({burst_frac[i]:.0%} burst)', ha='left', va='center', fontsize=9.5,
            color=col)
ax.set_xlim(-0.35,1.7); ax.set_ylim(6.6,0.4)
ax.set_xticks([0,1]); ax.set_xticklabels(['Raw star rank','Credibility-adjusted rank'])
ax.set_yticks(range(1,7)); ax.set_ylabel('Rank')
ax.set_title('Discounting inorganic bursts reshuffles the leaderboard  (synthetic data)', loc='left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.text(0,-0.13,
        'Repos with >20% of stars from flagged bursts (red) drop sharply once burst stars are discounted; organically-grown '
        'repos (green) rise to fill the gap.\nThe adjustment turns a gameable vanity metric into a credibility-weighted one.',
        transform=ax.transAxes, fontsize=8.5, color=MUTE, va='top')
plt.tight_layout(); plt.savefig(f'{OUT}/signalgraph_ranking.png', bbox_inches='tight'); plt.close()
print('fig2 done')
