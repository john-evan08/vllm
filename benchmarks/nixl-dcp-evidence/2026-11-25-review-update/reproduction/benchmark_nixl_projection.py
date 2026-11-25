"""Temporary MLA projection benchmark; launch with torchrun.

No production selector is added. This harness overrides the Python helper's
threshold only while issuing/capturing an arm, then restores it. Timings include
projection, exchange, and combine. GPU buffers and inputs are synthetic.
"""

import argparse
import contextlib
import datetime
import gc
import hashlib
import json
import os
import socket
from pathlib import Path


@contextlib.contextmanager
def projection_order(dcp, order):
    original = dcp._V_UP_PROJ_FIRST_MIN_TOKENS
    if order == "before":
        dcp._V_UP_PROJ_FIRST_MIN_TOKENS = 0
    elif order == "after":
        dcp._V_UP_PROJ_FIRST_MIN_TOKENS = 1 << 30
    try:
        yield
    finally:
        dcp._V_UP_PROJ_FIRST_MIN_TOKENS = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--topology", choices=("mixed", "local"), default="mixed")
    p.add_argument("--heads-per-rank", nargs="+", type=int, default=[6, 8])
    p.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=[1, 4, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    )
    p.add_argument("--layers", type=int, default=64)
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--iterations", type=int, default=64)
    args = p.parse_args()
    if args.layers < 1 or args.iterations % args.layers or args.samples < 3:
        p.error("iterations must be a multiple of layers; at least three samples")
    if args.samples % (4 if args.topology == "local" else 3):
        p.error("samples must be a multiple of the number of arms")
    import torch
    import torch.distributed as dist
    import vllm._custom_ops  # noqa: F401
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.attention.ops import dcp
    from vllm.v1.attention.ops.nixl_dcp import NixlDCPA2AWorkspace

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == (8 if args.topology == "local" else 16), world
    group = GroupCoordinator(
        [list(range(world))], local_rank, "nccl", use_device_communicator=False
    )
    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname(), group=group.cpu_group)
    assert len(set(hosts)) == (1 if args.topology == "local" else 2)
    assert all(hosts.count(h) == 8 for h in set(hosts))
    remote_peers = sum(h != hosts[rank] for h in hosts)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"rank{rank}.jsonl"
    assert not path.exists(), path
    log = path.open("x", buffering=1)

    def record(value):
        log.write(json.dumps(value) + "\n")

    record(
        {
            "type": "provenance",
            "rank": rank,
            "hosts": hosts,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "args": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "source": {
                str(Path(m.__file__).name): hashlib.sha256(
                    Path(m.__file__).read_bytes()
                ).hexdigest()
                for m in (dcp,)
            },
            "nccl_threshold": dcp._V_UP_PROJ_FIRST_MIN_TOKENS,
            "scope": f"H100 {args.topology} DCP{world}; BF16 L512/V128; no full-model timing",
        }
    )
    arms = {"nccl": "auto", "nixl_before": "before", "nixl_after": "after"}
    if args.topology == "local":
        arms["p2p_after"] = "after"
    for heads in args.heads_per_rank:
        direct = (
            dcp.DirectDCPA2AWorkspace(
                group.device_group,
                device,
                max(args.tokens),
                heads,
                512,
                torch.bfloat16,
            )
            if args.topology == "local"
            else None
        )
        workspaces = {
            dim: NixlDCPA2AWorkspace(
                group, device, max(args.tokens), heads, dim, torch.bfloat16
            )
            for dim in (512, 128)
        }
        for dim, ws in workspaces.items():
            for peer in range(world):
                if peer == rank:
                    continue
                if hosts[peer] != hosts[rank]:
                    assert ws.peer_copy_paths[peer] == "nixl_put"
                    assert "rc_mlx5_gda" in ws.peer_transports[peer]
                else:
                    assert ws.peer_copy_paths[peer] == "mapped_p2p"
            record(
                {
                    "type": "workspace",
                    "heads": heads,
                    "dim": dim,
                    "bytes": ws.workspace_bytes,
                    "send_bytes": ws.send.numel(),
                    "receive_bytes": ws.receive.numel(),
                    "paths": ws.peer_copy_paths,
                    "transports": ws.peer_transports,
                }
            )
        weights = torch.randn(
            (args.layers, world * heads, 512, 128),
            device=device,
            dtype=torch.bfloat16,
            generator=torch.Generator(device=device).manual_seed(1200),
        )
        weights.div_(512**0.5)

        def nixl_combine(partial, lse, **kwargs):
            return workspaces[partial.shape[-1]].lse_reduce(partial, lse, **kwargs)

        for tokens in args.tokens:
            gen = torch.Generator(device=device).manual_seed(4200 + rank + tokens)
            partial = torch.randn(
                (tokens, world * heads, 512),
                device=device,
                dtype=torch.bfloat16,
                generator=gen,
            )
            lse = torch.randn(
                (world * heads, tokens),
                device=device,
                dtype=torch.float32,
                generator=gen,
            ).T
            lengths = torch.tensor(
                [int(rank != 0), 3], device=device, dtype=torch.int32
            )
            starts = torch.tensor(
                [0, min(2, tokens), tokens], device=device, dtype=torch.int32
            )
            if rank == 0:
                partial[: min(2, tokens)] = float("nan")
            outputs = {
                name: torch.empty(
                    (tokens, heads * 128), device=device, dtype=torch.bfloat16
                )
                for name in arms
            }
            expected = torch.empty_like(outputs["nccl"])

            def call(name, weight_index, output=None, reference=False):
                combine = (
                    None
                    if reference or name == "nccl"
                    else direct.lse_reduce
                    if name == "p2p_after"
                    else nixl_combine
                )
                with projection_order(dcp, arms[name]):
                    dcp.dcp_a2a_v_up_proj_lse_reduce(
                        partial,
                        lse,
                        weights[weight_index],
                        outputs[name] if output is None else output,
                        group,
                        seq_lens=lengths,
                        query_start_loc=starts,
                        combine=combine,
                    )

            for layer_count in (1, args.layers):
                graphs = {}
                for name in arms:
                    for i in range(args.layers):
                        call(name, i % layer_count)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    dist.barrier(group=group.cpu_group)
                    with torch.cuda.graph(graph):
                        for i in range(args.iterations):
                            call(name, i % layer_count)
                    graphs[name] = graph
                # Changing inputs catches stale payload/epoch bugs in both orders.
                correctness_checks = []
                for _ in range(3):
                    partial.mul_(0.9)
                    lse.add_(0.125 * (rank + 1))
                    for name, graph in graphs.items():
                        call(
                            name,
                            (args.iterations - 1) % layer_count,
                            output=expected,
                            reference=True,
                        )
                        graph.replay()
                        error = (
                            (outputs[name].float() - expected.float())
                            .abs()
                            .max()
                            .item()
                        )
                        if name == "p2p_after":
                            # Existing P2P test tolerance; its CUDA normalization
                            # differs from the NCCL/NIXL Triton combine arithmetic.
                            torch.testing.assert_close(
                                outputs[name].float(),
                                expected.float(),
                                rtol=3e-2,
                                atol=3e-2,
                                msg=f"{name}: same-order NCCL reference",
                            )
                            call(
                                name,
                                (args.iterations - 1) % layer_count,
                                output=expected,
                            )
                        torch.testing.assert_close(
                            outputs[name],
                            expected,
                            rtol=0,
                            atol=0,
                            msg=f"{name}: graph reference",
                        )
                        correctness_checks.append(
                            {
                                "arm": name,
                                "max_abs_vs_nccl": error,
                                "rtol": 3e-2 if name == "p2p_after" else 0,
                                "atol": 3e-2 if name == "p2p_after" else 0,
                                "exact_p2p_eager_replay": name == "p2p_after",
                            }
                        )
                    torch.cuda.synchronize()
                record(
                    {
                        "type": "correctness",
                        "heads": heads,
                        "tokens": tokens,
                        "layers": layer_count,
                        "exact_same_order_nccl": args.topology != "local",
                        "checks": correctness_checks,
                        "changing_graph_replays": 3,
                    }
                )
                # Balanced first-arm order; retain every rank's time.
                for sample in range(args.samples):
                    names = list(arms)
                    offset = (sample + args.tokens.index(tokens)) % len(names)
                    order = names[offset:] + names[:offset]
                    for name in order:
                        graphs[name].replay()
                        torch.cuda.synchronize()
                        dist.barrier(group=group.cpu_group)
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        graphs[name].replay()
                        end.record()
                        end.synchronize()
                        elapsed = start.elapsed_time(end) * 1000 / args.iterations
                        times = [None] * world
                        dist.all_gather_object(times, elapsed, group=group.cpu_group)
                        width = (
                            128
                            if arms[name] == "before"
                            or (
                                name == "nccl"
                                and tokens >= dcp._V_UP_PROJ_FIRST_MIN_TOKENS
                            )
                            else 512
                        )
                        record(
                            {
                                "type": "timing",
                                "heads": heads,
                                "tokens": tokens,
                                "layers": layer_count,
                                "arm": name,
                                "sample": sample,
                                "order": order,
                                "us_by_rank": times,
                                "max_rank_us": max(times),
                                "rdma_payload_bytes_per_rank": remote_peers
                                * tokens
                                * heads
                                * (width * 2 + 4),
                                "active_weight_bytes": layer_count
                                * heads
                                * 512
                                * 128
                                * 2
                                * (world if width == 128 else 1),
                                "comparison_resident_bytes": torch.cuda.memory_allocated(),
                            }
                        )
                for graph in graphs.values():
                    graph.reset()
                graphs.clear()
                del graph
                torch.cuda.synchronize()
                if rank == 0:
                    print(
                        f"PASS heads={heads} tokens={tokens} layers={layer_count}",
                        flush=True,
                    )
            partial = lse = outputs = expected = None
            call = None
        weights = None
        direct = None
        for ws in workspaces.values():
            ws.close()
        workspaces.clear()
        del ws
        gc.collect()
        torch.cuda.empty_cache()
    record({"type": "complete", "passed": True})
    log.close()
    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
