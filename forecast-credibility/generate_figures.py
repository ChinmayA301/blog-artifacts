import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'savefig.dpi': 150,
})

INK   = '#1A1A1A'
MUTE  = '#9AA0A6'
BLUE  = '#4C72B0'
ORANGE= '#DD8452'
RED   = '#C44E52'
GREEN = '#55A868'

OUT = 'figures'
import os
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# FIGURE 1 — Implied-CAGR waterfall
# Decompose the 3.2 -> 322 jump into the compounding growth steps.
# Goldman's actual path is front-loaded then back-loaded, so we show
# the geometric (constant-CAGR) decomposition as the reference build
# and annotate where Goldman's published path diverges.
# ---------------------------------------------------------------
base = 3.2
end  = 322.0
years = [2025, 2026, 2027, 2028, 2029, 2030]
r = (end / base) ** (1/5)                     # geometric growth factor
cagr = (r - 1) * 100
smooth = [base * r**i for i in range(6)]       # constant-CAGR path
goldman = [3.2, 15.6, 34.5, None, None, 322.0] # published path (partial)

# Waterfall: each bar is the increment added that year on the smooth path
increments = [smooth[i+1] - smooth[i] for i in range(5)]
fig, ax = plt.subplots(figsize=(11, 6.2))

running = base
# starting block
ax.bar(0, base, width=0.62, color=MUTE, edgecolor='white')
ax.text(0, base + 4, f'${base:.1f}B\n2025 base', ha='center', va='bottom', fontsize=9.5)

for i, inc in enumerate(increments):
    bottom = running
    ax.bar(i+1, inc, bottom=bottom, width=0.62, color=BLUE, edgecolor='white', alpha=0.9)
    # connector line
    ax.plot([i+0.31, i+1-0.31], [running, running], color=MUTE, lw=0.8, ls='--')
    ax.text(i+1, bottom + inc + 4, f'+${inc:.0f}B', ha='center', va='bottom', fontsize=9)
    running += inc

# endpoint block
ax.bar(6, end, width=0.62, color=RED, edgecolor='white', alpha=0.9)
ax.text(6, end + 4, f'${end:.0f}B\n2030 target', ha='center', va='bottom', fontsize=9.5, color=RED)

ax.set_xticks(range(7))
ax.set_xticklabels(['2025\nbase', '2026', '2027', '2028', '2029', '2030', '2030\ntarget'])
ax.set_ylabel('AI revenue ($B)')
ax.set_title('Implied-CAGR waterfall: what a 100× build actually requires',
             loc='left')
ax.set_ylim(0, 360)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:.0f}B'))

ax.text(0.0, -0.20,
        f'Constant-CAGR decomposition of Goldman\u2019s $3.2B\u2192$322B projection: {cagr:.0f}% per year, sustained five years.\n'
        f'Goldman\u2019s published path is front-loaded (+388% in 2026 to $15.6B) then defers the bulk to the unobservable out-years.',
        transform=ax.transAxes, fontsize=8.5, color=MUTE, va='top')

plt.tight_layout()
plt.savefig(f'{OUT}/forecast_cagr_waterfall.png', bbox_inches='tight')
plt.close()
print('fig1 done | CAGR =', round(cagr,1), '% | smooth path =', [round(s,1) for s in smooth])

# ---------------------------------------------------------------
# FIGURE 2 — Path comparison (Goldman front/back-loaded vs smooth CAGR)
# This is the "where does the mass sit" evidence chart.
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(years, smooth, marker='o', lw=2.4, color=BLUE, label='Constant-CAGR path (151%/yr)')

gx = [y for y, v in zip(years, goldman) if v is not None]
gv = [v for v in goldman if v is not None]
ax.plot(gx, gv, marker='s', lw=2.4, color=ORANGE, ls='-', label='Goldman published path')
# dashed interpolation across the unpublished 2028-29 segment
ax.plot([2027, 2030], [34.5, 322], color=ORANGE, lw=1.4, ls=':', alpha=0.8)

ax.annotate('front-loaded\n+388% year 1', xy=(2026, 15.6), xytext=(2026.05, 70),
            fontsize=8.5, color=ORANGE, ha='left',
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1))
ax.annotate('heavy lift deferred\ninto out-years', xy=(2029, 175), xytext=(2027.4, 250),
            fontsize=8.5, color=ORANGE, ha='left',
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1))

ax.set_ylabel('AI revenue ($B)')
ax.set_xlabel('Year')
ax.set_title('Same endpoint, different risk: where the forecast hides its mass', loc='left')
ax.legend(loc='upper left', frameon=True)
ax.set_ylim(0, 360)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:.0f}B'))
ax.text(0.0, -0.16,
        'A forecast whose growth is concentrated in the out-years is most sensitive to assumptions that cannot yet be tested.',
        transform=ax.transAxes, fontsize=8.5, color=MUTE, va='top')
plt.tight_layout()
plt.savefig(f'{OUT}/forecast_path_comparison.png', bbox_inches='tight')
plt.close()
print('fig2 done')

# ---------------------------------------------------------------
# FIGURE 3 — Monte Carlo terminal-revenue distribution
# Sample correlated price/utilisation/adoption multipliers around the
# headline, propagate to 2030 terminal revenue, show P10/P50/P90 vs $322B.
# This replaces the linear one-at-a-time sensitivity with a real
# distribution -- the methodological upgrade the article argues for.
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
N = 200_000

# Multiplicative factors on the headline endpoint. Means centered slightly
# below 1.0 to reflect the documented historical tendency of AI forecasts
# to overshoot (2017 cohort undershot realized by 6-8x => forecasts ran hot).
# Correlated because a soft-demand world hits price AND utilisation together.
mean = np.array([0.92, 0.90, 0.95])          # price, utilisation, adoption-timing
sd   = np.array([0.18, 0.20, 0.22])
# correlation: price & utilisation strongly co-move; adoption moderately
corr = np.array([[1.00, 0.65, 0.40],
                 [0.65, 1.00, 0.45],
                 [0.40, 0.45, 1.00]])
cov = np.outer(sd, sd) * corr
draws = rng.multivariate_normal(mean, cov, size=N)
draws = np.clip(draws, 0.05, None)            # no negative multipliers
terminal = end * draws.prod(axis=1)

p10, p50, p90 = np.percentile(terminal, [10, 50, 90])

fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(terminal, bins=140, range=(0, 700), color=BLUE, alpha=0.78, edgecolor='white', linewidth=0.2)
for val, lab, col in [(p10,'P10',GREEN),(p50,'P50 (median)',INK),(p90,'P90',GREEN)]:
    ax.axvline(val, color=col, ls='--', lw=1.6)
    ax.text(val, ax.get_ylim()[1]*0.96, f'{lab}\n${val:.0f}B', color=col,
            ha='center', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=col, lw=0.8))
ax.axvline(end, color=RED, lw=2.2)
ax.text(end, ax.get_ylim()[1]*0.55, f'  Headline\n  ${end:.0f}B', color=RED,
        ha='left', va='center', fontsize=9, fontweight='bold')

ax.set_xlabel('2030 terminal AI revenue ($B)')
ax.set_ylabel('Simulated frequency')
ax.set_title('Monte Carlo: the headline sits in the upper tail of its own assumptions', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 700)
share_below = (terminal < end).mean() * 100
ax.text(0.0, -0.16,
        f'200,000 draws over correlated price / utilisation / adoption-timing multipliers (means set below 1.0 to reflect the documented\n'
        f'tendency of AI forecasts to overshoot). {share_below:.0f}% of simulated outcomes fall below the $322B headline. Illustrative priors, not a prediction.',
        transform=ax.transAxes, fontsize=8.5, color=MUTE, va='top')
plt.tight_layout()
plt.savefig(f'{OUT}/forecast_monte_carlo.png', bbox_inches='tight')
plt.close()
print('fig3 done | P10/P50/P90 =', round(p10), round(p50), round(p90), '| % below headline =', round(share_below,1))
