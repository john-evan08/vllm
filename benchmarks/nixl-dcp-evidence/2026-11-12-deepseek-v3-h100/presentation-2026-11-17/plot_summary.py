"""Redraw the published fixed-length DeepSeek-V3 study without changing data."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter

LENGTHS = (8192, 65536, 122880)
CONCURRENCIES = (1, 4, 16, 32)
LABELS = {8192: "8K", 65536: "64K", 122880: "120K"}
COLORS = {8192: "#007F73", 65536: "#3775B8", 122880: "#BD7621"}
EXPECTED_SHA256 = "28e1ef4fc74d227217a6e18554266645b44789641be834a3eb27579c1f3f8d29"
INK = "#162C3F"
MUTED = "#526473"


def load_data(source, reference):
    """Check the published input and independently recompute all paired values."""
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise ValueError("CSV differs from the published 76128 evidence.")
    rows = list(csv.DictReader(raw.decode().splitlines()))
    expected = {
        (backend, repeat, length, concurrency)
        for backend in ("nccl", "nixl")
        for repeat in (1, 2)
        for length in LENGTHS
        for concurrency in CONCURRENCIES
    }
    indexed = {}
    for row in rows:
        key = (
            row["backend"],
            int(row["repetition"]),
            int(row["input_tokens"]),
            int(row["concurrency"]),
        )
        if key in indexed:
            raise ValueError("Duplicate measurement.")
        indexed[key] = float(row["throughput_tok_s_gpu"])
    if set(indexed) != expected or len(rows) != 48:
        raise ValueError("Missing or extra measured cells.")
    published = {
        (p["input_tokens"], p["concurrency"]): p
        for p in json.loads(reference.read_text())["paired"]
    }
    summary = []
    for length in LENGTHS:
        for concurrency in CONCURRENCIES:
            nccl = [indexed["nccl", r, length, concurrency] for r in (1, 2)]
            nixl = [indexed["nixl", r, length, concurrency] for r in (1, 2)]
            ratios = [b / a for a, b in zip(nccl, nixl)]
            ratio = math.sqrt(math.prod(ratios))
            original = published[length, concurrency]
            if not math.isclose(ratio, original["geomean_ratio"], rel_tol=1e-12):
                raise ValueError("Paired mean differs from the published result.")
            if any(
                not math.isclose(a, b, rel_tol=1e-12)
                for a, b in zip(ratios, original["repeat_ratios"])
            ):
                raise ValueError("Paired repeat differs from the published result.")
            summary.append(
                {
                    "input_tokens": length,
                    "concurrency": concurrency,
                    "nccl": nccl,
                    "nixl": nixl,
                    "ratios": ratios,
                    "gain_percent": 100 * (ratio - 1),
                    "repeat_gains_percent": [100 * (r - 1) for r in ratios],
                    "nccl_mean_tok_s_gpu": statistics.mean(nccl),
                    "nixl_mean_tok_s_gpu": statistics.mean(nixl),
                }
            )
    return summary


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E3E8EC", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=9, colors=MUTED)


def export(fig, output, name):
    for extension in ("png", "pdf", "svg"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=180, facecolor="white")
        if extension == "svg":
            # Keep generated path-data lines free of trailing whitespace in Git.
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def plot_gains(summary, output):
    fig = plt.figure(figsize=(12.8, 7.4))
    ax = fig.add_axes((0.095, 0.22, 0.87, 0.47))
    fig.text(
        0.07,
        0.945,
        "NIXL gains at low concurrency; close to NCCL at high concurrency",
        fontsize=19,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.07,
        0.902,
        "DeepSeek-V3 FP8  |  2 × 8 H100  |  TP16 / DCP16  |  Fixed-length synthetic prompts",
        fontsize=11,
        color=MUTED,
    )
    fig.legend(
        handles=[Patch(color=COLORS[n], label=f"{LABELS[n]} input") for n in LENGTHS],
        loc="center",
        bbox_to_anchor=(0.51, 0.85),
        ncol=3,
        frameon=False,
        fontsize=11,
        handlelength=1.4,
        columnspacing=3,
    )
    for left, heading, points in (
        (
            0.115,
            "LOWER CONCURRENCY · 1–4",
            [p for p in summary if p["concurrency"] <= 4],
        ),
        (
            0.60,
            "HIGHER CONCURRENCY · 16–32",
            [p for p in summary if p["concurrency"] >= 16],
        ),
    ):
        gains = [p["gain_percent"] for p in points]
        fig.text(left, 0.79, heading, fontsize=10.5, color=MUTED, weight="bold")
        fig.text(
            left,
            0.742,
            f"+{min(gains):.1f}% to +{max(gains):.1f}%",
            fontsize=22,
            weight="bold",
            color=INK,
        )
    positions = dict(zip(CONCURRENCIES, (0, 1, 2.8, 3.8)))
    ax.axvspan(-0.55, 1.55, color="#F2F8F7", zorder=-2)
    ax.axvspan(2.25, 4.35, color="#F5F7F9", zorder=-2)
    for point in summary:
        length = point["input_tokens"]
        x = positions[point["concurrency"]] + (LENGTHS.index(length) - 1) * 0.25
        mean = point["gain_percent"]
        repeats = point["repeat_gains_percent"]
        ax.bar(x, mean, width=0.19, color=COLORS[length], zorder=2)
        ax.plot([x, x], [min(repeats), max(repeats)], color=INK, lw=1.3, zorder=3)
        ax.scatter(
            [x, x],
            repeats,
            marker="o",
            s=23,
            facecolors="white",
            edgecolors=INK,
            linewidths=1,
            zorder=4,
        )
        ax.text(
            x,
            max(repeats) + 1.15,
            f"+{mean:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color=INK,
            weight="bold",
        )
    ax.axhline(0, color="#71818D", linewidth=1.1)
    ax.text(1.9, 0.9, "NCCL parity", ha="center", color=MUTED, fontsize=9)
    ax.set(
        xlim=(-0.55, 4.35),
        ylim=(-1, 33),
        xticks=list(positions.values()),
        xticklabels=[str(c) for c in CONCURRENCIES],
        yticks=range(0, 31, 5),
        xlabel="Offered concurrency (simultaneous requests)",
        ylabel="Output throughput gain over NCCL",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.xaxis.labelpad = 14
    ax.yaxis.labelpad = 13
    style_axis(ax)
    fig.text(
        0.07,
        0.108,
        "Bars: paired ratio geometric mean, expressed as % gain. Dots and lines: both repeats and their range; not a confidence interval.",
        fontsize=9.3,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.075,
        "All 12 workload points shown · 512 output tokens · Prefix cache off · Two repeats on one physical node pair",
        fontsize=9.3,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.042,
        "Full serving-path comparison, including prefill. Study 76128 · Source b1ba99ec247b · Measurements unchanged.",
        fontsize=9.3,
        color=MUTED,
    )
    export(fig, output, "throughput-gain")


def plot_absolute(summary, output):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.7))
    fig.subplots_adjust(left=0.08, right=0.975, top=0.69, bottom=0.25, wspace=0.32)
    fig.text(
        0.07,
        0.93,
        "Absolute throughput: NIXL and NCCL converge at higher concurrency",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.07,
        0.88,
        "DeepSeek-V3 FP8 · 2 × 8 H100 · Fixed-length synthetic prompts · 512 output tokens",
        fontsize=11,
        color=MUTED,
    )
    for ax, length in zip(axes, LENGTHS):
        points = [p for p in summary if p["input_tokens"] == length]
        for backend, color, marker in (
            ("nccl", "#677782", "s"),
            ("nixl", "#007F73", "o"),
        ):
            means = [statistics.mean(p[backend]) for p in points]
            lower = [m - min(p[backend]) for m, p in zip(means, points)]
            upper = [max(p[backend]) - m for m, p in zip(means, points)]
            ax.errorbar(
                range(4),
                means,
                yerr=[lower, upper],
                color=color,
                marker=marker,
                ms=6.5,
                lw=2,
                capsize=4,
                linestyle="--" if backend == "nccl" else "-",
                label=backend.upper(),
                markerfacecolor="white",
                zorder=3,
            )
            for x, p in enumerate(points):
                ax.scatter(
                    [x - 0.035, x + 0.035],
                    p[backend],
                    s=10,
                    color=color,
                    alpha=0.65,
                    zorder=4,
                )
        ax.set_title(
            f"{LABELS[length]} input",
            loc="left",
            color=INK,
            fontsize=13,
            weight="bold",
            pad=15,
        )
        peak = max(max(p[b]) for p in points for b in ("nccl", "nixl"))
        ax.set(
            xlim=(-0.2, 3.2),
            ylim=(0, peak * 1.15),
            xticks=range(4),
            xticklabels=[str(c) for c in CONCURRENCIES],
            xlabel="Offered concurrency",
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.labelpad = 12
        style_axis(ax)
    axes[0].set_ylabel("Output tokens / second / GPU", labelpad=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.52, 0.80),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    fig.text(
        0.07,
        0.115,
        "Points: arithmetic means. Whiskers: observed range of two runs; small dots: individual runs.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.075,
        "Each panel starts at zero; vertical scales differ by input length. Prefix cache off; one physical node pair.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.035,
        "All 48 measured cells retained. Aggregate throughput includes prefill; this is not an isolated transport benchmark.",
        fontsize=10,
        color=MUTED,
    )
    export(fig, output, "absolute-throughput")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = load_data(args.results, args.reference)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    plot_gains(summary, args.output)
    plot_absolute(summary, args.output)
    (args.output / "plotted-values.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "passed": True,
                "input_sha256": EXPECTED_SHA256,
                "measured_cells": 48,
                "paired_points": len(summary),
                "matched_published_ratios": True,
                "matplotlib": matplotlib.__version__,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
