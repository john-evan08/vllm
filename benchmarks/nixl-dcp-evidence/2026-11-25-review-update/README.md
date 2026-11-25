# NIXL DCP review evidence — November 25

Completed mixed DCP16 and local DCP8 projection/exchange/combine microbenchmarks,
historical Mooncake serving results, and bounded validation reports are included.
See [scope and reproduction notes](OVERVIEW.txt) and [checksums](manifest.json).

The implementation is commit 809dd16e952aa155b658d895e5f58929addc5f3b.
The history-only publication merge has exactly the same source tree.
GPU measurements retain their recorded source/overlay/runtime provenance.

- [Mixed DCP16 results](microbenchmarks/mixed16/README.md)
- [Local DCP8 results, including symmetric P2P](microbenchmarks/local8/README.md)
- [Historical Mooncake serving replay](historical-mooncake-76482/README.md)
- [Validation reports](validation)

**The final projection-order model benchmark is pending.** No partial or
rejected allocation is presented as a completed comparison. The PR stays a
paused draft; current evidence does not establish consistent NCCL improvement.

Copied report files retain their original preparation-date wording, including
the then-unpublished status and then-unselected production policy. The current
implementation adopts NCCL's eight-query-token heuristic; forced-order timing
does not directly measure that dynamic policy. AI assistance: OpenAI Codex.
