"""Figures for reporting the messenger-speech narrativity result.

1. Per-play line-position plot: P(narrative) against line number, with known
   messenger-speech ranges shaded, so the reader can see the score rise
   inside spans that were delimited independently of the model.
2. Pooled distribution comparison: P(narrative) for messenger-speech vs.
   other tragedy sentences, the visual counterpart to the Mann-Whitney test.

Recomputes the (play, sent_id, line, is_messenger_speech) table directly from
the treebank + JSON (cheap, no model needed) and merges it against the
already-classified csv/messenger_speech_narrativity.csv, so this doesn't
require re-running the classifier.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

from compare_messenger_speech_narrativity import (
    build_sentence_table,
    load_messenger_speeches,
    urn_to_stem,
)

ROOT_DIR = Path(__file__).parent.parent
RESULTS_CSV = ROOT_DIR / "csv" / "messenger_speech_narrativity.csv"
FIG_DIR = ROOT_DIR / "figures"

# Validated categorical pair (scripts/validate_palette.js, light mode: PASS).
COLOR_MESSENGER = "#2a78d6"
COLOR_OTHER = "#1baf7a"
COLOR_GRID = "#d8d6cd"
COLOR_TEXT = "#52514e"


def load_merged() -> pd.DataFrame:
    sentences = build_sentence_table()
    results = pd.read_csv(RESULTS_CSV)[["play", "sent_id", "p_narrative"]]
    sentences["sent_id"] = sentences["sent_id"].astype(str)
    results["sent_id"] = results["sent_id"].astype(str)
    return sentences.merge(results, on=["play", "sent_id"], how="inner")


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_GRID)
    ax.spines["bottom"].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT, labelsize=8)
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_line_position(df: pd.DataFrame, plays: list[dict]) -> None:
    titles = {urn_to_stem(p["urn"]): p["title"] for p in plays}
    speech_ranges = {urn_to_stem(p["urn"]): p["speeches"] for p in plays}

    play_order = sorted(df["play"].unique())
    n = len(play_order)
    ncols = 2
    nrows = -(-n // ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(11, 2.6 * nrows), sharey=True, constrained_layout=True
    )
    axes = axes.flatten()

    ROLLING_WINDOW = 7

    for ax, play in zip(axes, play_order):
        play_df = df[df["play"] == play].sort_values("line")
        for start, end in speech_ranges[play]:
            ax.axvspan(start, end, color=COLOR_MESSENGER, alpha=0.15, linewidth=0)

        # Individual sentences are too jagged/discontinuous to connect with a
        # line (adjacent clauses can jump from 0 to 1 with no real relation),
        # so raw values are a faint scatter; a rolling mean shows the local
        # trend, which is what actually reveals a span "lighting up".
        ax.scatter(
            play_df["line"],
            play_df["p_narrative"],
            s=5,
            c=[
                COLOR_MESSENGER if m else COLOR_OTHER
                for m in play_df["is_messenger_speech"]
            ],
            linewidths=0,
            alpha=0.35,
            zorder=2,
        )
        rolling = (
            play_df["p_narrative"]
            .rolling(ROLLING_WINDOW, center=True, min_periods=1)
            .mean()
        )
        ax.plot(
            play_df["line"],
            rolling,
            color=COLOR_TEXT,
            linewidth=1.3,
            zorder=3,
        )

        ax.set_title(titles.get(play, play), fontsize=9, color=COLOR_TEXT, loc="left")
        ax.set_ylim(-0.05, 1.05)
        style_axis(ax)

    for ax in axes[n:]:
        ax.set_visible(False)

    for ax in axes[::ncols]:
        ax.set_ylabel("P(narrative)", fontsize=8, color=COLOR_TEXT)
    for ax in axes[max(0, n - ncols) : n]:
        ax.set_xlabel("line number", fontsize=8, color=COLOR_TEXT)

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=COLOR_MESSENGER,
            label="messenger-speech line range / sentence",
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="", color=COLOR_OTHER, label="other sentence"
        ),
        plt.Line2D(
            [],
            [],
            color=COLOR_TEXT,
            linewidth=1.3,
            label=f"{ROLLING_WINDOW}-sentence rolling mean",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "P(narrative) by line position, with known messenger-speech ranges shaded",
        fontsize=11,
        color=COLOR_TEXT,
    )

    FIG_DIR.mkdir(exist_ok=True, parents=True)
    out_path = FIG_DIR / "line_position.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")


def plot_distribution(df: pd.DataFrame) -> None:
    messenger = df.loc[df["is_messenger_speech"], "p_narrative"]
    other = df.loc[~df["is_messenger_speech"], "p_narrative"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)

    parts = ax.violinplot(
        [other, messenger], positions=[0, 1], showmedians=True, widths=0.8
    )
    for body, color in zip(parts["bodies"], [COLOR_OTHER, COLOR_MESSENGER]):  # ty:ignore[not-iterable, invalid-argument-type]
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.55)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color(COLOR_TEXT)
        parts[key].set_linewidth(1)

    for x, series, color in [(0, other, COLOR_OTHER), (1, messenger, COLOR_MESSENGER)]:
        ax.scatter(
            [x],
            [series.mean()],
            marker="D",
            s=28,
            color=color,
            edgecolor=COLOR_TEXT,
            linewidth=0.6,
            zorder=4,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"other tragedy\n(n={len(other)})", f"messenger speech\n(n={len(messenger)})"],
        fontsize=9,
        color=COLOR_TEXT,
    )
    ax.set_ylabel("P(narrative)", fontsize=9, color=COLOR_TEXT)
    ax.set_ylim(-0.05, 1.05)
    style_axis(ax)
    ax.xaxis.grid(False)

    _, mw_p = mannwhitneyu(messenger, other, alternative="greater")
    ax.text(
        0.5,
        1.0,
        f"diamond = mean; Mann-Whitney p = {mw_p:.2g}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color=COLOR_TEXT,
    )
    ax.set_title(
        "P(narrative): messenger-speech vs. other tragedy sentences",
        fontsize=10.5,
        color=COLOR_TEXT,
        pad=14,
    )

    FIG_DIR.mkdir(exist_ok=True, parents=True)
    out_path = FIG_DIR / "distribution.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main():
    df = load_merged()
    plays = load_messenger_speeches()
    plot_line_position(df, plays)
    plot_distribution(df)


if __name__ == "__main__":
    main()
