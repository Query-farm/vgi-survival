"""Presentation helpers; cohort transformation and estimators stay in visible SQL."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def theme():
    """Apply consistent readable chart defaults."""
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 150,
            "text.parse_math": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def finish(fig, name, title, subtitle):
    """Display and save a figure with the synthetic source and time units."""
    fig.suptitle(title, x=0.09, y=0.98, ha="left", fontsize=17, fontweight="bold", color="#243e32")
    fig.text(0.09, 0.92, subtitle, ha="left", fontsize=10.5, color="#526b5b")
    fig.text(
        0.09,
        0.025,
        "Source: 24 invented device follow-up records · no observed devices or clinical data\n"
        "vgi-survival / lifelines 0.30.3 through Haybarn / VGI · days since synthetic time zero",
        fontsize=9,
        color="#526b5b",
    )
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.2, top=0.79, hspace=0.2)
    directory = Path("validation/charts")
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", bbox_inches="tight")
    plt.show()
