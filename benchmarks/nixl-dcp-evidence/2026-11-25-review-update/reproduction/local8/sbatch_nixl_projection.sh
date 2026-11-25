#!/bin/bash
#SBATCH --job-name=nixl-projection
#SBATCH --partition=GAIA
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
set -euo pipefail
PROJ=${DCP_EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:?}}
source "$PROJ/scripts/workspace_env.sh"
: "${NIXL_PROJECTION_HARNESS:?}" "${NIXL_OVERLAY_SHA256:?}"
ROOT=/mnt/lustre/jkarcenty/dcp_nixl
IMAGE="$ROOT/images/review-20261119-qualify-77036/runtime.sqsh"
export CONTAINER_SHA256=3877af650ab46630c7dcfbe2ba7b2e7d619501012f2d25768ffa05a2b7f02b7e
printf '%s  %s\n' "$CONTAINER_SHA256" "$IMAGE" | sha256sum --check
export OUTDIR="$ROOT/run_artifacts/nixl-projection-$SLURM_JOB_ID"
RUNDIR="$PROJ/.runscripts/$SLURM_JOB_ID"
mkdir "$OUTDIR" "$RUNDIR"
ln -s "$OUTDIR" "$PROJ/results/nixl-projection-$SLURM_JOB_ID"
cp "$NIXL_PROJECTION_HARNESS/"* "$RUNDIR/"
(cd "$RUNDIR" && sha256sum --check harness.sha256) > "$OUTDIR/harness-verification.txt"
printf 'job=%s\nnodes=%s\nbase_commit=f0311094893c3bdee51d3bb74b3d23459a78c0db\nimage=%s\nimage_sha256=%s\noverlay_sha256=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$IMAGE" "$CONTAINER_SHA256" "$NIXL_OVERLAY_SHA256" > "$OUTDIR/provenance.txt"
export SCRATCH="/raid/users/$USER/nixl-projection-$SLURM_JOB_ID"
export NIXL_SOURCE_IMAGE="$IMAGE"
env -u TMPDIR srun --kill-on-bad-exit=1 bash -c '
  set -euo pipefail
  mkdir -p "$SCRATCH/image" "$SCRATCH/tmp" "$SCRATCH/ctmp" "$SCRATCH/enroot/tmp" "$SCRATCH/enroot/data" "$SCRATCH/enroot/cache"
  cp "$NIXL_SOURCE_IMAGE" "$SCRATCH/image/runtime.sqsh"
  printf "%s  %s\n" "$CONTAINER_SHA256" "$SCRATCH/image/runtime.sqsh" | sha256sum --check
'
export TMPDIR="$SCRATCH/tmp" ENROOT_TEMP_PATH="$SCRATCH/enroot/tmp"
export ENROOT_DATA_PATH="$SCRATCH/enroot/data" ENROOT_CACHE_PATH="$SCRATCH/enroot/cache"
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
srun --kill-on-bad-exit=1 --container-image="$SCRATCH/image/runtime.sqsh" --no-container-mount-home \
  --container-mounts="$PROJ:/work,$ROOT:$ROOT,$SCRATCH/ctmp:/ctmp,$TMPDIR:$TMPDIR,/dev/infiniband:/dev/infiniband" \
  --container-workdir=/opt/dcp/vllm \
  bash "/work/.runscripts/$SLURM_JOB_ID/nixl_projection_inner.sh" "/work/.runscripts/$SLURM_JOB_ID" "$OUTDIR"
touch "$OUTDIR/job-complete"
