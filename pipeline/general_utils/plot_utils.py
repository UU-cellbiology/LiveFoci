import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats
from itertools import combinations

   
##############

# Core plot functions
    
##############


def plot_AT(shifted_time, plot_samples, remove_final_frames=0, ax=None):
    """
    Plot mean active tracks per cell over time.

    Parameters
    ----------
    shifted_time : 1-D array
        Time axis in hours (length = num_frames).
    plot_samples : list of dicts
        Each dict must have keys: active_tracks (2-D array, cells × frames),
        name, color, linestyle.
    remove_final_frames : int
        Number of tail frames to drop (avoids artefact from min_track_length).
    ax : matplotlib Axes, optional
    """
    if ax is None:
        ax = plt.gca()

    end = len(shifted_time) - remove_final_frames if remove_final_frames > 0 else len(shifted_time)

    for s in plot_samples:
        mean_at = np.nanmean(s["active_tracks"], 0)
        ax.plot(shifted_time[:end], mean_at[:end],
                color=s["color"], label=s["name"],
                linestyle=s.get("linestyle", "solid"))

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Active tracks per cell")
    ax.legend(loc="upper right")


def plot_hist(bins, plot_samples, log=False, ax=None):
    """
    Normalised dwell-time histogram.

    Parameters
    ----------
    bins : array-like
        Bin edges in minutes.
    plot_samples : list of dicts
        Each dict must have: track_lengths (1-D array), name, color, alpha,
        facecolor, edgecolor.
    log : bool
        Use log scale on the y-axis.
    ax : matplotlib Axes, optional
    """
    if ax is None:
        ax = plt.gca()

    for s in plot_samples:
        tl = np.asarray(s["track_lengths"])
        if len(tl) == 0:
            continue
        weights = np.ones_like(tl, dtype=float) / len(tl)
        ax.hist(tl, bins=bins, weights=weights, label=s["name"], facecolor=s.get("facecolor", "none"), edgecolor=s.get("edgecolor", s["color"]),
                alpha=s.get("alpha", 0.7), histtype="bar")

    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    else:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    ax.set_ylabel("Percentage [%]")
    ax.set_xlabel("Dwell time [min]")
    ax.legend()


def plot_violin(plot_samples, max_bin, ax=None):
    """
    Violin plot of dwell-time distributions.

    Parameters
    ----------
    plot_samples : list of dicts
        Each dict must have: track_lengths (1-D array), name, color,
        and optionally hatch.
    max_bin : int
        Upper y-axis limit in minutes.
    ax : matplotlib Axes, optional
    """
    if ax is None:
        ax = plt.gca()

    data    = [np.asarray(s["track_lengths"]) for s in plot_samples]
    labels  = [s["name"] for s in plot_samples]
    colors  = [s["color"] for s in plot_samples]
    hatches = [s.get("hatch", "") for s in plot_samples]

    # filter out empty conditions
    valid = [(d, l, c, h) for d, l, c, h in zip(data, labels, colors, hatches) if len(d) > 0]
    if not valid:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    data, labels, colors, hatches = zip(*valid)

    n         = len(data)
    positions = np.arange(n) * 1.5 + 1
    width     = 0.7

    vp = ax.violinplot(data, positions=positions, widths=width, showmeans=True, showmedians=False, showextrema=False, points=max_bin * 2)

    vp["cmeans"].set_linewidth(2.5)
    vp["cmeans"].set_color(list(colors))

    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor("none")
        body.set_edgecolor(colors[i])
        body.set_linewidth(1.5)
        if hatches[i]:
            body.set_hatch(hatches[i])
        body.set_alpha(1.0)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dwell time [min]")
    ax.set_ylim(0, max_bin)
    ax.set_xlabel("Condition")
    ax.set_xlim(positions[0] - 1, positions[-1] + 1)


##############

# Statistical tests
    
##############

def _mean_statistic(x, y):
    return np.mean(x) - np.mean(y)


def _bootstrap_mean_difference(x, y, n_bootstrap=5000, ci=95):
    rng = np.random.default_rng()
    x, y = np.asarray(x), np.asarray(y)
    boot = np.array([
        np.mean(rng.choice(x, size=len(x), replace=True)) -
        np.mean(rng.choice(y, size=len(y), replace=True))
        for _ in range(n_bootstrap)
    ])
    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)
    return float(np.mean(x) - np.mean(y)), float(lo), float(hi), boot


def run_statistical_tests(plot_samples, n_resamples=5000, alternative="two-sided"):
    """
    Run pairwise statistical tests on dwell-time distributions.

    Tests performed for each pair:
      - Permutation test on the difference of means
      - Bootstrap 95 % confidence interval for the mean difference
      - Mann-Whitney U test

    Parameters
    ----------
    plot_samples : list of dicts
        Each dict must have keys: name (str), track_lengths (1-D array-like).
    n_resamples : int
        Number of resamples for the permutation test.
    alternative : str
        One of 'two-sided', 'less', 'greater'.

    Returns
    -------
    str
        Formatted results string.
    """
    lines = []

    for s1, s2 in combinations(plot_samples, 2):
        d1 = np.asarray(s1["track_lengths"], dtype=float)
        d2 = np.asarray(s2["track_lengths"], dtype=float)

        lines.append(f"{'─'*60}")
        lines.append(f"  {s1['name']}  vs  {s2['name']}")
        lines.append(f"  Number of tracks: {len(d1):>6}  vs  {len(d2)}")
        lines.append(f"  Mean dwell time: {np.mean(d1):>6.1f} min  vs  {np.mean(d2):.1f} min")
        lines.append(f"  Mean difference: {np.mean(d2) - np.mean(d1):+.1f} min")

        res = stats.permutation_test((d2, d1), _mean_statistic, vectorized=False, alternative=alternative, 
                                     permutation_type="independent", n_resamples=n_resamples)
        lines.append(f"  Permutation test p-value:  {res.pvalue:.4g}")

        _, lo, hi, _ = _bootstrap_mean_difference(d2, d1)
        lines.append(f"  Bootstrap 95 % CI:         [{lo:.2f}, {hi:.2f}] min")

        u, p_mwu = stats.mannwhitneyu(d2, d1, use_continuity=False, alternative=alternative)
        lines.append(f"  Mann-Whitney U: {u:.0f},  p-value: {p_mwu:.4g}")
        lines.append("")

    if not lines:
        return "Need at least two conditions to compare."

    return "\n".join(lines)
