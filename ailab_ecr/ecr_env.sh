# Shared settings for the ECR AI-Lab scripts. Sourced, not run.
#
# Expected layout on the cluster (override with environment variables if yours differs):
#
#   $ECR_HOME/                 default: $HOME/ecr
#     ecr.sif                  the Singularity image you build in step 2
#     *.sh *.py                this bundle
#     ECR/                     the git clone (so $ECR_SRC is $ECR_HOME/ECR/src_emo)
#
# All of ECR's paths ("data/...", "save/...", "log/...") are relative to the current
# directory, so every script below runs with the working directory set to $ECR_SRC.

ECR_HOME="${ECR_HOME:-$HOME/ecr}"
ECR_SIF="${ECR_SIF:-$ECR_HOME/ecr.sif}"
ECR_SRC="${ECR_SRC:-$ECR_HOME/ECR/src_emo}"
# BUNDLE_DIR is set by the calling script before it sources this file.
ECR_BUNDLE="${ECR_BUNDLE:-${BUNDLE_DIR:-$ECR_HOME}}"

# WANDB_MODE is set in ecr.def, but repeat it here: ECR imports wandb unconditionally and
# an online attempt on a compute node would block. SINGULARITYENV_* passes into the image.
export SINGULARITYENV_WANDB_MODE=offline
# Keep the host's ~/.local/lib python packages out of the container.
export SINGULARITYENV_PYTHONNOUSERSITE=1

# The command is "singularity" on the compute nodes, but some sites install it as
# "apptainer", and it is absent from the front-end PATH entirely -- which is why every
# script here runs on a compute node via srun/sbatch, never on the front-end.
SING="${SING:-$(command -v singularity 2>/dev/null || command -v apptainer 2>/dev/null || echo singularity)}"

ecr_check_layout() {
  local ok=0
  [ -f "$ECR_SIF" ] || { echo "MISSING: $ECR_SIF  (build it: step 2 in README.md)"; ok=1; }
  [ -d "$ECR_SRC" ] || { echo "MISSING: $ECR_SRC  (git clone ECR: step 1)"; ok=1; }
  return $ok
}
