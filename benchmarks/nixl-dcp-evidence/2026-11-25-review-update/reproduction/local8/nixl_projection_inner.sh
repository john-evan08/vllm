#!/bin/bash
set -euo pipefail
RUNDIR=${1:?}; OUTDIR=${2:?}
export VIRTUAL_ENV=/opt/dcp/.venv
export PATH="$VIRTUAL_ENV/bin:/opt/dcp/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64
unset NIXL_PLUGIN_DIR UCX_MODULE_DIR UCX_HOME NIXL_HOME
export PYTHONDONTWRITEBYTECODE=1 NCCL_NET=IB NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_DIRECT_DCP_A2A=0 VLLM_USE_NIXL_DCP_A2A=0
export VLLM_USE_DIRECT_DCP_Q_GATHER=0 VLLM_USE_DIRECT_DCP_KV_GATHER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0 OMP_NUM_THREADS=1
export TMPDIR=/ctmp/tmp VLLM_CACHE_ROOT=/ctmp/vllm_cache TRITON_CACHE_DIR=/ctmp/triton_cache
export XDG_CACHE_HOME=/ctmp/cache CUDA_CACHE_PATH=/ctmp/cuda-cache
mkdir -p "$TMPDIR"
rank=${SLURM_PROCID:?}
topology=${NIXL_PROJECTION_TOPOLOGY:-mixed}
nodes=${SLURM_NNODES:?}
if [[ $topology == local ]]; then
  test "$nodes" = 1
else
  test "$topology" = mixed
  test "$nodes" = 2
fi
test "$(cat /opt/dcp/vllm-native-source-sha256)" = d2c1689feea7ed573a0ed5c19c9205d165be5c04049bb0e17f65ff8c0e7bbbde
test "$(cat /opt/dcp/vllm-source-sha256)" = 5e8ee1c60352c705172db7a7050262557d05aa6c8452c7b3b694d3fe68c576f7
printf '%s  %s\n' "$NIXL_OVERLAY_SHA256" "$RUNDIR/overlay.tar" | sha256sum --check
tar -xf "$RUNDIR/overlay.tar" -C /opt/dcp/vllm
(cd /opt/dcp/vllm && sha256sum --check "$RUNDIR/overlay-files.sha256") > "$OUTDIR/overlay-verify-node$rank.txt"
hostname > "$OUTDIR/hostname-node$rank.txt"
nvidia-smi topo -m > "$OUTDIR/topology-node$rank.txt"
nvidia-smi -q > "$OUTDIR/nvidia-smi-node$rank.txt"
cp /opt/dcp/runtime-packages.txt "$OUTDIR/runtime-packages-node$rank.txt"
nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.sm,utilization.gpu,clocks_throttle_reasons.active,power.draw \
  --format=csv --loop=1 --filename="$OUTDIR/telemetry-node$rank.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT
barrier() {
  touch "$OUTDIR/$1-node$rank"
  local deadline=$((SECONDS + 900))
  until [[ -f "$OUTDIR/$1-node0" && ( $nodes == 1 || -f "$OUTDIR/$1-node1" ) ]]; do
    if (( SECONDS >= deadline )); then echo "Barrier timeout: $1" >&2; exit 1; fi
    sleep 1
  done
}
uv run --no-project --python .venv/bin/python .venv/bin/python -B "$RUNDIR/check_gpu_health.py" "$OUTDIR/health-node$rank" --seconds 120 > "$OUTDIR/health-node$rank.log" 2>&1
barrier health-passed
if [[ $rank == 0 && -n ${NIXL_REGRESSION_EVIDENCE:-} ]]; then
  grep -qx "overlay_sha256=$NIXL_OVERLAY_SHA256" "$NIXL_REGRESSION_EVIDENCE/provenance.txt"
  grep -q '^66 passed, 25 deselected' "$NIXL_REGRESSION_EVIDENCE/cpu-tests.log"
  uv run --no-project --python .venv/bin/python .venv/bin/python - "$NIXL_REGRESSION_EVIDENCE/direct-regression.xml" <<'PYXML'
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
cases = root.findall('.//testcase')
assert len(cases) == 2
assert not root.findall('.//failure') + root.findall('.//error') + root.findall('.//skipped')
PYXML
  cp "$NIXL_REGRESSION_EVIDENCE/cpu-tests.log" "$NIXL_REGRESSION_EVIDENCE/direct-regression.log" "$NIXL_REGRESSION_EVIDENCE/direct-regression.xml" "$OUTDIR/"
  printf '%s\n' "$NIXL_REGRESSION_EVIDENCE" > "$OUTDIR/regression-evidence-reused-from.txt"
elif [[ $rank == 0 ]]; then
  uv run --no-project --python .venv/bin/python .venv/bin/python -B -m pytest --noconftest -q -p no:cacheprovider \
    tests/distributed/test_dcp_a2a.py tests/distributed/test_dcp_direct_a2a_lse_reduce.py \
    -k 'not test_distributed and not rejects_models' > "$OUTDIR/cpu-tests.log" 2>&1
  uv run --no-project --python .venv/bin/python .venv/bin/python -B -m pytest --noconftest -q -p no:cacheprovider \
    tests/distributed/test_dcp_direct_a2a_lse_reduce.py \
    -k 'test_distributed_direct_a2a_matches_reference and transposed' \
    --junitxml="$OUTDIR/direct-regression.xml" > "$OUTDIR/direct-regression.log" 2>&1
fi
barrier regression-passed
mkdir -p "$OUTDIR/functional" "$OUTDIR/benchmark"
if [[ $topology == local ]]; then
if [[ -n ${NIXL_LOCAL_FUNCTIONAL_EVIDENCE:-} ]]; then
  grep -qx "overlay_sha256=$NIXL_OVERLAY_SHA256" "$NIXL_LOCAL_FUNCTIONAL_EVIDENCE/provenance.txt"
  uv run --no-project --python .venv/bin/python .venv/bin/python - "$NIXL_LOCAL_FUNCTIONAL_EVIDENCE/functional/local-nixl.xml" <<'PYXML'
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
assert len(root.findall('.//testcase')) == 8
assert not root.findall('.//failure') + root.findall('.//error') + root.findall('.//skipped')
PYXML
  cp "$NIXL_LOCAL_FUNCTIONAL_EVIDENCE/functional/local-nixl.xml" "$OUTDIR/functional/"
  cp "$NIXL_LOCAL_FUNCTIONAL_EVIDENCE/functional-node0.log" "$OUTDIR/"
  printf '%s\n' "$NIXL_LOCAL_FUNCTIONAL_EVIDENCE" > "$OUTDIR/local-functional-reused-from.txt"
else
uv run --no-project --python .venv/bin/python .venv/bin/python -B -m pytest --noconftest -q -p no:cacheprovider \
  tests/distributed/test_dcp_direct_a2a_lse_reduce.py \
  -k 'test_distributed_nixl_a2a_matches_reference' \
  --junitxml="$OUTDIR/functional/local-nixl.xml" > "$OUTDIR/functional-node$rank.log" 2>&1
fi
else
uv run --no-project --python .venv/bin/python .venv/bin/python -B -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=8 --node-rank="$rank" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  "$RUNDIR/validate_nixl_review.py" --output "$OUTDIR/functional" --topology mixed --gpus-per-node 8 \
  > "$OUTDIR/functional-node$rank.log" 2>&1
fi
barrier functional-passed
benchmark_args=()
if [[ $topology == local ]]; then
  benchmark_args=(--topology local --heads-per-rank 12 16 --samples 8)
fi
uv run --no-project --python .venv/bin/python .venv/bin/python -B -m torch.distributed.run \
  --nnodes="$nodes" --nproc-per-node=8 --node-rank="$rank" \
  --master-addr="$MASTER_ADDR" --master-port="$((MASTER_PORT + 1))" \
  "$RUNDIR/benchmark_nixl_projection.py" --output "$OUTDIR/benchmark" "${benchmark_args[@]}" \
  > "$OUTDIR/benchmark-node$rank.log" 2>&1
barrier benchmark-passed
kill "$sampler"
wait "$sampler" 2>/dev/null || true
trap - EXIT
uv run --no-project --python .venv/bin/python .venv/bin/python -B "$RUNDIR/check_dsv3_telemetry.py" \
  "$OUTDIR/telemetry-node$rank.csv" "$OUTDIR/sustained-health-node$rank.json" > "$OUTDIR/sustained-health-node$rank.log" 2>&1
barrier complete
