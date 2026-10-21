#!/bin/bash
#SBATCH --job-name=ecr-smoke
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
#
# T1 steps 3 and 5 in one job: prove the GPU works for the operations ECR actually uses,
# then run ECR's own model code for one training step and check seed reproducibility.
# Needs no data and no checkpoints. Takes about a minute.
#
#   sbatch ecr_smoke.sh
#
# Expect, from gpu_check.py:  CUDA available: True / sm_89 / matmul ok / RGCN ok / torch_sparse ok
# Expect, from ecr_smoke.py:  same seed reproducible: True | different seed differs: True
set -uo pipefail
# Slurm runs a COPY of this script from its spool dir, so $0's directory is not the
# bundle. SLURM_SUBMIT_DIR is where you ran sbatch from; $0 covers a direct "bash" run.
BUNDLE_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
[ -f "$BUNDLE_DIR/ecr_env.sh" ] || { echo "cannot find ecr_env.sh next to $BUNDLE_DIR -- submit from the bundle directory, or set ECR_BUNDLE"; exit 1; }
source "$BUNDLE_DIR/ecr_env.sh"
ecr_check_layout || exit 1

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo

echo "=== gpu_check.py ==="
"$SING" exec --nv --pwd "$ECR_BUNDLE" "$ECR_SIF" python gpu_check.py
echo

# Note: sm_89 may be absent from torch's compiled arch list and still work -- Ada runs
# sm_86 cubins, and the matmul/RGCN lines are the real evidence. Don't fail on that alone.

echo "=== ecr_smoke.py (ECR's own model, one step, no data) ==="
"$SING" exec --nv --pwd "$ECR_BUNDLE" "$ECR_SIF" python ecr_smoke.py "$ECR_SRC"
