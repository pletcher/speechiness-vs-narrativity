from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress, mannwhitneyu

from ALL_DRAMA_compare_messenger_speeches import (
    build_sentence_table,
    load_messenger_speeches,
    urn_to_stem,
)

ROOT_DIR = Path(__file__).parent.parent
RESULTS_CSV = ROOT_DIR / "csv" / "ALL_DRAMA_messenger_speech_narrativity.csv"
FIG_DIR = ROOT_DIR / "figures"

# Validated categorical pair (scripts/validate_palette.js, light mode: PASS).
COLOR_MESSENGER = "#799ccd"
COLOR_OTHER = "#cd9279"
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


def slugify(title: str) -> str:
    return title.lower().replace(" ", "_")


ROLLING_WINDOW = 7


def plot_line_position(df: pd.DataFrame, plays: list[dict]) -> None:
    titles = {urn_to_stem(p["urn"]): f"{p['dramatist']}, {p['title']}" for p in plays}
    speech_ranges = {urn_to_stem(p["urn"]): p["speeches"] for p in plays}

    FIG_DIR.mkdir(exist_ok=True, parents=True)

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
        plt.Line2D(
            [],
            [],
            color=COLOR_TEXT,
            linewidth=1.3,
            linestyle="--",
            label="linear trend",
        ),
    ]

    for play in sorted(df["play"].unique()):
        play_df = df[df["play"] == play].sort_values("line")
        title = titles.get(play, play)

        fig, ax = plt.subplots(figsize=(9, 3.2), constrained_layout=True)

        for start, end in speech_ranges[play]:
            ax.axvspan(start, end, color=COLOR_MESSENGER, alpha=0.15, linewidth=0)

        # Individual sentences are too discontinuous to connect with a
        # line (adjacent clauses can jump from 0 to 1 with no real relation);
        # a rolling mean shows the local trend, which is what actually
        # reveals a span "lighting up".
        ax.scatter(
            play_df["line"],
            play_df["p_narrative"],
            s=8,
            c=[
                COLOR_MESSENGER if m else COLOR_OTHER
                for m in play_df["is_messenger_speech"]
            ],
            linewidths=0,
            alpha=0.4,
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

        trend = linregress(play_df["line"], play_df["p_narrative"])
        ax.plot(
            play_df["line"],
            trend.intercept + trend.slope * play_df["line"],
            color=COLOR_TEXT,
            linewidth=1.3,
            linestyle="--",
            zorder=3,
        )
        ax.text(
            0.99,
            0.99,
            f"trend p = {trend.pvalue:.2g}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color=COLOR_TEXT,
        )

        ax.set_title(title, fontsize=11, color=COLOR_TEXT, loc="left")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(narrative)", fontsize=9, color=COLOR_TEXT)
        ax.set_xlabel("line number", fontsize=9, color=COLOR_TEXT)
        style_axis(ax)

        ax.legend(
            handles=handles,
            loc="upper center",
            ncol=1,
            frameon=False,
            fontsize=7,
            bbox_to_anchor=(1.18, 1.0),
        )

        out_path = FIG_DIR / f"ALL_DRAMA_line_position_{play}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")


def plot_distribution(df: pd.DataFrame) -> None:
    messenger = df.loc[df["is_messenger_speech"], "p_narrative"]
    other = df.loc[~df["is_messenger_speech"], "p_narrative"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)

    parts = ax.violinplot(
        [other, messenger], positions=[0, 1], showmedians=True, widths=0.8
    )
    for body, color in zip(parts["bodies"], [COLOR_OTHER, COLOR_MESSENGER]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.55)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color(COLOR_TEXT)
        parts[key].set_linewidth(1)

    for x, series, color in [(0, other, COLOR_OTHER), (1, messenger, COLOR_MESSENGER)]:
        mean = series.mean()
        ax.scatter(
            [x],
            [mean],
            marker="D",
            s=28,
            color=color,
            edgecolor=COLOR_TEXT,
            linewidth=0.6,
            zorder=4,
        )
        ax.annotate(
            f"{mean:.3f}",
            (x, mean),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=COLOR_TEXT,
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
    out_path = FIG_DIR / "ALL_DRAMA_distribution.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")


def plot_change_over_time(df: pd.DataFrame, plays: list[dict]) -> None:
    year_by_play = {urn_to_stem(p["urn"]): p["year"] for p in plays}
    dramatist_by_play = {urn_to_stem(p["urn"]): p["dramatist"] for p in plays}
    title_by_play = {urn_to_stem(p["urn"]): p["title"] for p in plays}

    per_play = (
        df.groupby(["play", "is_messenger_speech"])["p_narrative"].mean().reset_index()
    )

    per_play["year"] = per_play["play"].map(year_by_play)
    per_play["dramatist"] = per_play["play"].map(dramatist_by_play)
    per_play["title"] = per_play["play"].map(title_by_play)
    per_play["category"] = per_play["is_messenger_speech"].map(
        {True: "messenger speech", False: "other tragedy"}
    )
    per_play = per_play.dropna(subset=["year"])

    category_scale = alt.Scale(
        domain=["messenger speech", "other tragedy"],
        range=[COLOR_MESSENGER, COLOR_OTHER],
    )

    points = (
        alt.Chart(per_play)
        .mark_circle(size=70, opacity=0.85)
        .encode(
            x=alt.X(
                "year:Q",
                title="approx. production year (BCE shown as negative)",
                scale=alt.Scale(domain=[-500, -400]),
            ),
            y=alt.Y(
                "p_narrative:Q",
                title="mean P(narrative)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "category:N", scale=category_scale, legend=alt.Legend(title=None)
            ),
            tooltip=[
                alt.Tooltip("title:N", title="play"),
                alt.Tooltip("dramatist:N"),
                alt.Tooltip("year:Q"),
                alt.Tooltip("p_narrative:Q", title="mean P(narrative)", format=".3f"),
            ],
        )
    )

    trend = points.transform_regression(
        "year", "p_narrative", groupby=["category"], method="linear"
    ).mark_line(size=2)

    chart = (
        (points + trend)
        .properties(width=560, height=340, title="Mean P(narrative) by play over time")
        .configure_axis(
            gridColor=COLOR_GRID,
            domainColor=COLOR_GRID,
            tickColor=COLOR_GRID,
            labelColor=COLOR_TEXT,
            titleColor=COLOR_TEXT,
            labelFontSize=9,
            titleFontSize=9,
        )
        .configure_title(color=COLOR_TEXT, fontSize=11, anchor="start")
        .configure_legend(labelColor=COLOR_TEXT, labelFontSize=9)
        .configure_view(strokeWidth=0)
    )

    FIG_DIR.mkdir(exist_ok=True, parents=True)
    out_path = FIG_DIR / "ALL_DRAMA_change_over_time.png"
    chart.save(out_path, scale_factor=2)
    print(f"Wrote {out_path}")


def main():
    df = load_merged()
    plays = load_messenger_speeches()
    plot_line_position(df, plays)
    plot_distribution(df)
    plot_change_over_time(df, plays)


if __name__ == "__main__":
    main()
