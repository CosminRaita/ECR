#!/bin/bash
#SBATCH --job-name=ecr-eval
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
#
# Evaluates ECR's RELEASED recommender checkpoint in test-only mode: no training, one
# forward pass per example, the full catalogue ranked. This is the reproduction check.
#
#   sbatch ecr_eval.sh
#
# Measured locally (RTX 4060, same sm_89 generation as the L4) at eval batch 4:
#   17.6 min, peak GPU 1.45 GiB allocated / 1.70 reserved, peak host RAM 6.09 GB.
# An L4 has 24 GB, so batch 32 fits comfortably; --mem=32G is generous for a 6 GB job.
#
# Eval batch size is 32 ON PURPOSE. ECR computes AUC within each batch and averages, so
# the reported AUC depends on this number -- see auc_batch_size.md. 32 is what the paper
# used, so this run is the apples-to-apples comparison.
#
# Expected (paper's ECR row, Table 1 / our local run):
#   recall@1 ~0.048   recall@10 ~0.218   recall@50 ~0.428
#   recall_true@1 ~0.054   @10 ~0.234   @50 ~0.451
#   auc ~0.540 at batch 32  (~0.596 if you lower it to 4 -- not a bug, see the note above)
set -uo pipefail
# Slurm runs a COPY of this script from its spool dir, so $0's directory is not the
# bundle. SLURM_SUBMIT_DIR is where you ran sbatch from; $0 covers a direct "bash" run.
BUNDLE_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
[ -f "$BUNDLE_DIR/ecr_env.sh" ] || { echo "cannot find ecr_env.sh next to $BUNDLE_DIR -- submit from the bundle directory, or set ECR_BUNDLE"; exit 1; }
source "$BUNDLE_DIR/ecr_env.sh"
ecr_check_layout || exit 1

cd "$ECR_SRC"
# evaluate_rec.py opens save/redial_rec/rec.json without creating the folder, and would
# otherwise crash when the evaluator is constructed, before any work is done.
mkdir -p save/redial_rec log save/eval_tmp

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# --prompt_encoder points at the released stage-2 weights.
# --output_dir is passed explicitly: it DEFAULTS to data/saved/rec, which is where those
#   released weights live, and train_rec.py creates <output_dir>/best inside it.
"$SING" exec --nv --pwd "$ECR_SRC" "$ECR_SIF" python "$ECR_BUNDLE/run_ecr.py" train_rec.py \
  --dataset redial_gen \
  --test \
  --prompt_encoder data/saved/rec/ \
  --output_dir save/eval_tmp \
  --n_prefix_rec 10 \
  --per_device_eval_batch_size 32 \
  --per_device_train_batch_size 4 \
  --context_max_length 200 \
  --prompt_max_length 200 \
  --entity_max_length 32 \
  --num_warmup_steps 530 \
  --learning_rate 1e-4 \
  --like_score 2.0 \
  --dislike_score 1.0 \
  --notsay_score 0.5 \
  --weighted_loss \
  --nei_mer \
  --use_sentiment \
  --seed 8

echo
# Patch 3 reopens this file in "w" mode every run, so keep a job-tagged copy before
# the next job truncates it.
cp save/redial_rec/auc_dump.jsonl "save/redial_rec/auc_dump_${SLURM_JOB_ID:-local}.jsonl" 2>/dev/null && echo "dump kept as save/redial_rec/auc_dump_${SLURM_JOB_ID:-local}.jsonl"
echo "=== AUC against evaluation batch size, recomputed from this run's scores ==="
# Patch 3 logged one line per example; this needs no GPU and takes seconds.
"$SING" exec --pwd "$ECR_SRC" "$ECR_SIF" python "$ECR_BUNDLE/auc_sweep.py" \
  save/redial_rec/auc_dump.jsonl
