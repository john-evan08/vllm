"""Validate and plot the temporary local or mixed projection comparison."""

import argparse
import collections
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not (args.run / "job-complete").exists():
        raise ValueError("Job completion marker missing")
    provenance = json.loads(
        (args.run / "benchmark/rank0.jsonl").read_text().splitlines()[0]
    )
    world = len(provenance["hosts"])
    nodes = len(set(provenance["hosts"]))
    local = nodes == 1
    arms = ["nccl", "nixl_before", "nixl_after"]
    if local:
        arms.append("p2p_after")
    for node in range(nodes):
        for name in (
            f"health-node{node}/summary.json",
            f"sustained-health-node{node}.json",
        ):
            assert json.loads((args.run / name).read_text())["passed"], name
    ranks = [
        [
            json.loads(line)
            for line in (args.run / f"benchmark/rank{rank}.jsonl")
            .read_text()
            .splitlines()
        ]
        for rank in range(world)
    ]
    assert all(rows[-1] == {"type": "complete", "passed": True} for rows in ranks)
    config = ranks[0][0]["args"]
    expected_cells = len(config["heads_per_rank"]) * len(config["tokens"]) * 2
    canonical = [r for r in ranks[0] if r["type"] == "timing"]
    for rank, rows in enumerate(ranks):
        assert sum(r["type"] == "correctness" for r in rows) == expected_cells
        if local:
            for row in (r for r in rows if r["type"] == "correctness"):
                assert len(row["checks"]) == 3 * len(arms)
                for check in row["checks"]:
                    if check["arm"] == "p2p_after":
                        assert check["exact_p2p_eager_replay"]
                        assert check["rtol"] == check["atol"] == 3e-2
                    else:
                        assert check["max_abs_vs_nccl"] == 0
        timing = [r for r in rows if r["type"] == "timing"]
        assert len(timing) == expected_cells * len(arms) * config["samples"]
        for actual, expected in zip(timing, canonical):
            assert {
                k: v for k, v in actual.items() if k != "comparison_resident_bytes"
            } == {
                k: v for k, v in expected.items() if k != "comparison_resident_bytes"
            }, f"Rank {rank} timing agreement failed"
        for ws in (r for r in rows if r["type"] == "workspace"):
            assert collections.Counter(ws["paths"]) == {
                "self": 1,
                "mapped_p2p": 7,
                **({} if local else {"nixl_put": 8}),
            }
    grouped = collections.defaultdict(list)
    for row in canonical:
        grouped[row["heads"], row["tokens"], row["layers"], row["arm"]].append(
            row["max_rank_us"]
        )
    summary = []
    for (heads, tokens, layers, arm), samples in sorted(grouped.items()):
        nccl = statistics.median(grouped[heads, tokens, layers, "nccl"])
        median = statistics.median(samples)
        summary.append(
            {
                "heads_per_rank": heads,
                "tokens": tokens,
                "rotating_layers": layers,
                "arm": arm,
                "median_us": median,
                "min_us": min(samples),
                "max_us": max(samples),
                "nccl_over_arm": nccl / median,
                "samples_us": samples,
            }
        )
    workspace = [r for r in ranks[0] if r["type"] == "workspace"]
    memory = []
    for heads in config["heads_per_rank"]:
        for name, dim, multiplier in (
            ("nixl_before", 128, world),
            ("nixl_after", 512, 1),
        ):
            ws = next(r for r in workspace if r["heads"] == heads and r["dim"] == dim)
            memory.append(
                {
                    "heads_per_rank": heads,
                    "arm": name,
                    "capacity_tokens": max(config["tokens"]),
                    "workspace_bytes": ws["bytes"],
                    "weight_bytes_per_layer": heads * multiplier * 512 * 128 * 2,
                    "note": "Fixed-order policy requirements; benchmark keeps both layouts and gathered weights resident. Excludes graph pools, intermediates and NIXL internal allocations.",
                }
            )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "passed": True,
                "config": config,
                "provenance": (args.run / "provenance.txt").read_text(),
                "rows": summary,
                "memory": memory,
            },
            indent=2,
        )
        + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"nccl": "#555555", "nixl_before": "#157fca", "nixl_after": "#dc6e24"}
    labels = {
        "nccl": "NCCL (existing policy)",
        "nixl_before": "NIXL: project before",
        "nixl_after": "NIXL: project after",
    }
    if local:
        colors["p2p_after"] = "#208349"
        labels["p2p_after"] = "P2P: project after (unchanged)"
    fig, axes = plt.subplots(
        len(config["heads_per_rank"]), 2, figsize=(12, 8), squeeze=False
    )
    for i, heads in enumerate(config["heads_per_rank"]):
        for j, layers in enumerate((1, config["layers"])):
            ax = axes[i][j]
            for arm in colors:
                points = sorted(
                    (
                        r
                        for r in summary
                        if r["heads_per_rank"] == heads
                        and r["rotating_layers"] == layers
                        and r["arm"] == arm
                    ),
                    key=lambda r: r["tokens"],
                )
                x = [r["tokens"] for r in points]
                ax.plot(
                    x,
                    [r["median_us"] for r in points],
                    "o-",
                    color=colors[arm],
                    label=labels[arm],
                    markersize=3,
                )
                ax.fill_between(
                    x,
                    [r["min_us"] for r in points],
                    [r["max_us"] for r in points],
                    color=colors[arm],
                    alpha=0.12,
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(
                f"{heads * world} heads; {'one repeated layer' if layers == 1 else str(layers) + ' rotating layers'}"
            )
            ax.set_xlabel("Query tokens per invocation")
            ax.set_ylabel("Projection + exchange + combine (µs)")
            ax.grid(True, alpha=0.2)
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2 if local else 3)
    topology = "NVLink only" if local else "mixed NVLink/RDMA"
    fig.suptitle(
        f"{nodes} node(s) × 8 H100 · {topology} · DCP{world} · BF16 · L512 → V128\nMedian of {config['samples']} max-rank samples; shading shows sample range"
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(args.output / "projection-comparison.png", dpi=180)
    fig.savefig(args.output / "projection-comparison.svg")
    text = [
        f"# {topology} DCP{world} projection comparison",
        "",
        "Correctness, topology, rank records and sampled GPU health gates passed.",
        "",
        f"Each timing includes projection, exchange and combine. {config['samples']} samples on {nodes} physical H100 node(s); this is not full-model performance. The 64-layer rotation probes weight-cache sensitivity and does not guarantee cold cache.",
        "",
        "NCCL uses the existing eight-token policy. The NIXL arms force one order for every batch size. The production policy has **not** been selected.",
        "",
    ]
    if local:
        max_p2p_error = max(
            check["max_abs_vs_nccl"]
            for rows in ranks
            for row in rows
            if row["type"] == "correctness"
            for check in row["checks"]
            if check["arm"] == "p2p_after"
        )
        text += [
            f"Unchanged P2P uses its existing rtol=atol=0.03 numerical tolerance against same-order NCCL, plus exact graph-versus-eager replay equality. Maximum observed absolute projected-output difference from NCCL: {max_p2p_error:.8g}. NIXL remains exact against same-order NCCL.",
            "",
        ]
    for heads in config["heads_per_rank"]:
        text += [
            f"## {heads * world} heads, {config['layers']} rotating layers",
            "",
            "| Tokens | NCCL µs | NIXL before µs | NIXL after µs |"
            + (" P2P after µs |" if local else ""),
            "| ---: | ---: | ---: | ---: |" + (" ---: |" if local else ""),
        ]
        for tokens in config["tokens"]:
            values = [
                statistics.median(grouped[heads, tokens, config["layers"], arm])
                for arm in colors
            ]
            text.append(
                f"| {tokens} | " + " | ".join(f"{v:.2f}" for v in values) + " |"
            )
        text.append("")
    text += [
        "## Memory requirements",
        "",
        "Fixed-order registered workspaces at the tested capacity; weights shown per layer. Actual benchmark residency includes both layouts and the full weight bank, so it is not a measurement of either deployed policy's total memory.",
        "",
        "| Heads | NIXL order | Workspace MiB | Weights MiB/layer |",
        "| ---: | --- | ---: | ---: |",
    ]
    for m in memory:
        text.append(
            f"| {m['heads_per_rank'] * world} | {m['arm']} | {m['workspace_bytes'] / 2**20:.2f} | {m['weight_bytes_per_layer'] / 2**20:.2f} |"
        )
    text += [
        "",
        "Raw per-rank records, full sample ranges and provenance are retained. Graphs: `projection-comparison.png` and `.svg`. No PR update or final policy selection was made.",
    ]
    (args.output / "README.md").write_text("\n".join(text) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "cells": expected_cells,
                "timings": len(canonical),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
