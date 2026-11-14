"""Verified synthetic inputs and presentation for the survival notebooks."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from vgi_notebooks import finish

GREEN = "#247F70"
ORANGE = "#C77842"
BLUE = "#397F91"


def configure(value, mode="cutoff"):
    """Validate the visible parameter and fixture bytes before running SQL."""
    valid = isinstance(value, int) and not isinstance(value, bool)
    valid = valid and (1 <= value <= 22 if mode == "cutoff" else value in (5, 10))
    if not valid:
        raise ValueError("Use integer follow-up 1–22, or load increment 5 or 10.")
    manifest = json.loads(Path("fixtures/provenance.json").read_text())
    assert hashlib.sha256(Path("fixtures/followup.csv").read_bytes()).hexdigest() == manifest["sha256"]
    second = json.loads(Path("fixtures/load_provenance.json").read_text())
    assert hashlib.sha256(Path("fixtures/load_followup.csv").read_bytes()).hexdigest() == second["sha256"]
    return manifest


def plot_curve(curve, cutoff):
    """Contrast product-limit and fixed-denominator calculations."""
    fig, ax = plt.subplots(figsize=(11.3, 7.0))
    ax.step(curve.time, curve.survival, where="post", lw=2.4, color=GREEN, label="Kaplan–Meier")
    ax.fill_between(
        curve.time, curve.ci_lower, curve.ci_upper, step="post", color=GREEN, alpha=0.13, label="95% pointwise bounds"
    )
    ax.step(
        curve.time,
        curve.original_denominator,
        where="post",
        ls="--",
        lw=2,
        color=ORANGE,
        label="1 − observed failures / original N",
    )
    marks = curve.loc[curve.censored > 0]
    ax.scatter(marks.time, marks.survival, marker="+", s=90, color="#243E32", zorder=4, label="One or more censored")
    ax.axhline(0.5, color="#647A6B", lw=1, ls=":")
    ax.set(xlabel="Follow-up time (days)", ylabel="Estimated event-free fraction", xlim=(0, cutoff), ylim=(0, 1.04))
    ax.grid(axis="y")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.07, 0.82), ncol=2, fontsize=9)
    finish(
        fig,
        "Censoring changes the denominator after each exit",
        f"Administrative cutoff: {cutoff} days · estimates stop at recorded follow-up",
        "24 invented devices · released VGI Survival v0.4.0 · no field or clinical observations\n"
        "Confidence bounds are estimator output for the synthetic input, not calibrated population evidence.",
        name="01-survival",
    )


def plot_risk(curve, availability, cutoff):
    """Show risk sets and both kinds of exit without treating censoring as failure."""
    fig, (a, b) = plt.subplots(2, 1, figsize=(11.3, 8.5), sharex=True, gridspec_kw={"hspace": 0.4})
    a.plot(curve.time, curve.at_risk_sql, "o-", color=BLUE)
    a.set(ylabel="Devices at risk", ylim=(0, 26))
    a.grid(axis="y")
    exits = curve.loc[curve.time > 0]
    b.bar(exits.time, exits.failures, width=0.65, color=ORANGE, label="Observed failures")
    b.bar(exits.time, exits.censored, bottom=exits.failures, width=0.65, color=BLUE, label="Right-censored")
    b.set(xlabel="Follow-up time (days)", ylabel="Exits at this time", xlim=(-0.3, cutoff + 0.7))
    b.yaxis.get_major_locator().set_params(integer=True)
    fig.legend(*b.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.07, 0.82), ncol=2, fontsize=9)
    finish(
        fig,
        "The later curve rests on a smaller risk set",
        f"{int(availability.iloc[0].devices)} devices at time zero · {len(curve) - 1} observed exit times",
        "Invented device follow-up · released VGI Survival v0.4.0\n"
        "Risk sets are just before each exit time; connecting markers are only a visual guide.",
        name="02-risk-sets",
    )


def plot_groups(curves):
    """Keep group curves on their own observed support."""
    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    for group, color in [("Below 50%", GREEN), ("50% or above", ORANGE)]:
        f = curves.loc[curves.load_group == group].sort_values("time")
        ax.step(f.time, f.survival, where="post", color=color, lw=2.2, label=group)
        ax.fill_between(f.time, f.ci_lower, f.ci_upper, step="post", color=color, alpha=0.12)
    ax.set(xlabel="Follow-up time (days)", ylabel="Estimated event-free fraction", ylim=(0, 1.04))
    ax.grid(axis="y")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.07, 0.82), ncol=2, fontsize=9)
    finish(
        fig,
        "Group curves describe the fixed observations",
        "Load groups stay fixed · shading: 95% pointwise confidence bounds",
        "80 simulated devices · fixed-seed exponential event times with independent observation stops\n"
        "Released VGI Survival v0.4.0 · group curves do not establish a causal load effect.",
        name="03-load-groups",
    )


def plot_ratios(fit, step):
    """Label every ratio with its covariate increment on a log scale."""
    f = fit.iloc[0]
    values = [
        (f.hazard_ratio, f.ci_lower, f.ci_upper),
        (f.hr_per_ten, f.lower_per_ten, f.upper_per_ten),
        (f.reference_hr, f.reference_lower, f.reference_upper),
    ]
    labels = [f"Selected fit · per {step} points", "Converted · per 10 points", "Reference fit · per 10 points"]
    fig, ax = plt.subplots(figsize=(11.0, 6.1))
    for i, (value, lo, hi) in enumerate(values):
        ax.errorbar(
            value,
            i,
            xerr=[[value - lo], [hi - value]],
            fmt="o",
            ms=9,
            capsize=4,
            color=ORANGE if i == 0 else GREEN,
            lw=2,
        )
        ax.text(hi * 1.025, i, f"{value:.3f} [{lo:.3f}, {hi:.3f}]", va="center", fontsize=10)
    ax.axvline(1, color="#939B9A", ls="--", lw=1)
    ax.set_yticks(range(3), labels)
    ax.set(
        xscale="log",
        xlabel="Hazard ratio with 95% interval (log scale)",
        xlim=(min(0.95, min(v[1] for v in values) * 0.95), max(v[2] for v in values) * 1.40),
        ylim=(2.6, -0.6),
    )
    ax.set_xticks([1.0, 1.2, 1.5, 2.0], ["1.0", "1.2", "1.5", "2.0"])
    ax.minorticks_off()
    ax.grid(axis="x", alpha=0.4)
    finish(
        fig,
        "The increment is part of the hazard ratio",
        f"One load unit = {step} percentage points · the ten-point contrast is unchanged",
        "Same 80 simulated devices · released VGI Survival v0.4.0 · one continuous load covariate\n"
        "Relative instantaneous hazard under a Cox model; not an event probability or causal estimate.",
        name="04-load-ratios",
    )
