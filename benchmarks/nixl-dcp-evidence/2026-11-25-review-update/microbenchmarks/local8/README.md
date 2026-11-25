# NVLink only DCP8 projection comparison

Correctness, topology, rank records and sampled GPU health gates passed.

Each timing includes projection, exchange and combine. 8 samples on 1 physical H100 node(s); this is not full-model performance. The 64-layer rotation probes weight-cache sensitivity and does not guarantee cold cache.

NCCL uses the existing eight-token policy. The NIXL arms force one order for every batch size. The production policy has **not** been selected.

Unchanged P2P uses its existing rtol=atol=0.03 numerical tolerance against same-order NCCL, plus exact graph-versus-eager replay equality. Maximum observed absolute projected-output difference from NCCL: 0.015625. NIXL remains exact against same-order NCCL.

## 96 heads, 64 rotating layers

| Tokens | NCCL µs | NIXL before µs | NIXL after µs | P2P after µs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 25.12 | 20.77 | 18.29 | 24.54 |
| 4 | 30.37 | 21.65 | 21.12 | 26.22 |
| 7 | 30.33 | 22.21 | 22.25 | 26.85 |
| 8 | 31.51 | 22.52 | 23.15 | 27.19 |
| 16 | 33.83 | 24.59 | 22.72 | 29.66 |
| 32 | 34.61 | 27.20 | 33.19 | 34.37 |
| 64 | 38.86 | 31.59 | 53.49 | 44.29 |
| 128 | 52.30 | 47.98 | 101.98 | 79.49 |
| 256 | 83.38 | 83.65 | 203.15 | 185.12 |
| 512 | 128.25 | 162.77 | 386.27 | 404.81 |
| 1024 | 223.30 | 310.30 | 754.03 | 957.45 |
| 2048 | 395.43 | 585.37 | 1482.18 | 2021.02 |

## 128 heads, 64 rotating layers

| Tokens | NCCL µs | NIXL before µs | NIXL after µs | P2P after µs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 28.53 | 21.96 | 18.99 | 24.88 |
| 4 | 29.82 | 22.97 | 21.87 | 26.22 |
| 7 | 31.83 | 23.91 | 21.94 | 27.22 |
| 8 | 33.27 | 24.23 | 21.53 | 27.85 |
| 16 | 34.53 | 27.87 | 29.98 | 30.67 |
| 32 | 37.42 | 30.04 | 40.22 | 36.84 |
| 64 | 45.50 | 39.59 | 68.85 | 50.86 |
| 128 | 62.76 | 62.97 | 145.77 | 92.03 |
| 256 | 93.81 | 106.59 | 282.93 | 235.23 |
| 512 | 161.29 | 211.07 | 544.68 | 515.08 |
| 1024 | 283.08 | 406.43 | 1074.70 | 1274.42 |
| 2048 | 505.22 | 776.42 | 2122.88 | 2679.59 |

## Memory requirements

Fixed-order registered workspaces at the tested capacity; weights shown per layer. Actual benchmark residency includes both layouts and the full weight bank, so it is not a measurement of either deployed policy's total memory.

| Heads | NIXL order | Workspace MiB | Weights MiB/layer |
| ---: | --- | ---: | ---: |
| 96 | nixl_before | 105.98 | 12.00 |
| 96 | nixl_after | 393.41 | 1.50 |
| 128 | nixl_before | 138.06 | 16.00 |
| 128 | nixl_after | 520.28 | 2.00 |

Raw per-rank records, full sample ranges and provenance are retained. Graphs: `projection-comparison.png` and `.svg`. No PR update or final policy selection was made.
