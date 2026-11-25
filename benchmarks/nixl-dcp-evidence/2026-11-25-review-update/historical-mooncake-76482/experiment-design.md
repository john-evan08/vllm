# DeepSeek-V3 public Mooncake trace, H100 / RDMA

Compare production NCCL DCP with mixed NIXL DCP on two healthy eight-H10080GB
nodes, TP16/DCP16, DP1/PP1/PCP1, EP enabled. Other collectives retain NCCL.
Reuse the November 11 pinned FP8 checkpoint and b1ba99ec runtime; no new image.
NIXL uses mapped copies locally and device puts over RDMA for remote peers.

## Fixed workload

Use the public trace linked from the August 7 DCP blog, ai-dynamo/dynamo commit
66a95d5aa3b5d238fd94b6f8243c4e479cb7242c, file
recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl.
Source SHA256 f20d3f2bc83dd1306cda659fbe34e7c4d85ca5497626c98bc0b1c4d2211379d0.
Filter ONCE by input_length + output_length <= 131072: 3250/3541 retained.
Never truncate prompts or requested output. Preserve line bytes, hash IDs and
source order. The immutable manifest records retained/excluded source indexes.

Native AIperf 0.12.0 Mooncake loader, block size512, sequential sampling, seed42;
pinned DeepSeek-V3 tokenizer, completions endpoint without special tokens.
The all-record prompt audit checks actual tokenized context, expected input and
output counts, prompt-text and token-ID hashes. Every measured replay must cover
all retained records exactly once with zero errors/cancellations/context skips.
This is a length/prefix-sharing trace replay, not semantic model accuracy or
exact reproduction of the blog's model, hardware, scheduler or workload mix.
The filtered trace has 6,969,000 requested output tokens per complete replay.
This file contains no arrival schedule. Offered concurrency is closed loop,
with replacement requests after completions; the dispatch sequence is fixed,
but completion order and resulting cache reuse can vary with scheduling.
Report observed hit rates and admission rather than assuming they are identical.

## Serving and comparisons

Same 16-GiB/rank BF16 KV budget, 16384 batched-token limit, max64 sequences,
FlashMLA/DeepGEMM, piecewise graph sizes1/2/4/8/16/32/64. Enable prefix caching
in both backends. Warm with the first c trace prompts and32 output tokens;
drain and successfully reset prefix cache before every measurement.

Only c16 and c64; two repeats per backend, fresh server per pass in ABBA order:
NCCL1, NIXL1, NIXL2, NCCL2. Reverse concurrency order in repeat2. The complete
identical file is replayed at every point. No backend-specific tuning.
Capture throughput, interactivity, TTFT/latency, KV usage, actual running/waiting
requests, prefix-hit/query deltas and preemptions. Preemptions are retained.
Health gates reject hardware slowdown for the allocation, not selected cells.

Rerun mixed DCP16 functional/graph checks on the selected pair. Retain the
established nine generation checks and logprob bounds in one initialized model,
production NCCL/NIXL/NCCL-restoration, with cache reset between arms. Trace
batch16/64 dispatch. Fresh unmodified servers provide separate pilots (first128
trace requests at c64, explicitly outside final timing) and the full comparison.

Report two repeats on one physical pair as limited evidence. NCCL projects
before merge and NIXL merges latent outputs; this is the full serving-path
comparison, not equal-byte communication microbenchmarking. Registered NIXL
workspace costs remain explicit. Keep previous synthetic results linked, with
one compact figure/table in the PR. Do not mix study generations.
