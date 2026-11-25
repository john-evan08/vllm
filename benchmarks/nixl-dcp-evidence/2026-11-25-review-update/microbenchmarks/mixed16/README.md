# Mixed DCP16 projection comparison

Correctness, topology, rank records and sampled GPU health gates passed.

Each timing includes projection, exchange and combine. Six samples on one physical H100 node pair; this is not full-model performance. The 64-layer rotation probes weight-cache sensitivity and does not guarantee cold cache.

NCCL uses the existing eight-token policy. The NIXL arms force one order for every batch size. The production policy has **not** been selected.

## 96 heads, 64 rotating layers

| Tokens | NCCL µs | NIXL before µs | NIXL after µs |
| ---: | ---: | ---: | ---: |
| 1 | 66.33 | 51.77 | 50.40 |
| 4 | 73.20 | 53.76 | 54.42 |
| 7 | 129.99 | 53.71 | 59.19 |
| 8 | 68.96 | 52.73 | 60.52 |
| 16 | 77.24 | 57.04 | 70.39 |
| 32 | 92.28 | 64.03 | 89.36 |
| 64 | 123.90 | 75.64 | 126.17 |
| 128 | 137.72 | 101.85 | 233.62 |
| 256 | 193.18 | 160.11 | 438.90 |
| 512 | 296.53 | 287.35 | 830.73 |
| 1024 | 497.89 | 520.46 | 1525.36 |
| 2048 | 882.40 | 985.65 | 2773.51 |

## 128 heads, 64 rotating layers

| Tokens | NCCL µs | NIXL before µs | NIXL after µs |
| ---: | ---: | ---: | ---: |
| 1 | 60.87 | 53.31 | 47.35 |
| 4 | 83.78 | 57.02 | 54.99 |
| 7 | 88.24 | 54.02 | 64.74 |
| 8 | 72.96 | 55.08 | 63.98 |
| 16 | 87.24 | 59.35 | 74.30 |
| 32 | 103.64 | 71.06 | 103.14 |
| 64 | 123.19 | 87.38 | 165.87 |
| 128 | 166.59 | 126.88 | 317.22 |
| 256 | 226.29 | 204.94 | 574.30 |
| 512 | 368.74 | 377.72 | 1072.44 |
| 1024 | 636.50 | 684.12 | 1952.98 |
| 2048 | 1125.55 | 1266.05 | 3663.59 |

## Memory requirements

Fixed-order registered workspaces at the tested capacity; weights shown per layer. Actual benchmark residency includes both layouts and the full weight bank, so it is not a measurement of either deployed policy's total memory.

| Heads | NIXL order | Workspace MiB | Weights MiB/layer |
| ---: | --- | ---: | ---: |
| 96 | nixl_before | 113.97 | 12.00 |
| 96 | nixl_after | 400.94 | 0.75 |
| 128 | nixl_before | 147.59 | 16.00 |
| 128 | nixl_after | 528.81 | 1.00 |

Raw per-rank records, full sample ranges and provenance are retained. Graphs: `projection-comparison.png` and `.svg`. No PR update or final policy selection was made.
