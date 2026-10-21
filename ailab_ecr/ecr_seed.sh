#!/bin/bash
#SBATCH --job-name=ecr-seed
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#
# One real training run of train_rec.py at a given seed, for the seed-robustness study.
#
#   sbatch ecr_seed.sh 8          # ECR's own seed and its full recipe (5 epochs)
#   sbatch ecr_seed.sh 1          # seed 1, same recipe
#   sbatch ecr_seed.sh 2 3        # seed 2, shortened to 3 epochs
#
# Several of these can run AT THE SAME TIME. That is the whole point of this script
# existing separately from ecr_train.sh: evaluate_rec.py opens "save/redial_rec/rec.json"
# and (patch 3) "save/redial_rec/auc_dump.jsonl" by fixed relative paths, both in "w"
# mode, so two concurrent jobs sharing a working directory would overwrite each other's
# logs. So each job gets its own working directory under $ECR_HOME/runs/, holding
# symlinks to ECR's sources and to data/ (which the dataset code only ever reads) and a
# real save/ of its own.
#
# The flags below are ECR's README command for train_rec.py, verbatim. Two of them are
# easy to drop by accident and both change what is being trained:
#   --num_train_epochs 5            their value (not 10)
#   --gradient_accumulation_steps 8 effective batch 128, not 16. It also sets the step
#                                   count: ceil(2117/8) = 265 optimiser steps per epoch,
#                                   so 5 epochs = 1,325 steps, which is what makes their
#                                   --num_warmup_steps 530 a sane number.
#
# Cost, from job 915118 (1.15 s/step at train batch 16, 15.30 GiB GPU allocated):
#   accumulation does not change the number of forward/backward passes, so an epoch is
#   still ~2,117 passes ~= 41 min, plus a valid+test pass of ~8 min. There is NO baseline
#   evaluation before training: train_rec.py:271 guards that path with `if args.test`
#   and the training loop is its `else`. So 5 epochs = 5 rounds ~= 3 h 45 min.
#   Measured on seed 8 (job 915277): 8.26 s per optimiser step, 1,325 steps.
#
# NOT yet measured: peak GPU memory with accumulation on. Training already sits at 15.3 of
# 23 GiB, and train_rec.py:421 passes retain_graph=True, so watch the first job's reported
# peak before launching more.
set -uo pipefail
BUNDLE_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
[ -f "$BUNDLE_DIR/ecr_env.sh" ] || { echo "cannot find ecr_env.sh next to $BUNDLE_DIR -- submit from the bundle directory, or set ECR_BUNDLE"; exit 1; }
source "$BUNDLE_DIR/ecr_env.sh"
ecr_check_layout || exit 1

SEED="${1:-}"
EPOCHS="${2:-5}"
[ -n "$SEED" ] || { echo "usage: sbatch ecr_seed.sh <seed> [epochs]"; exit 1; }

JOB="${SLURM_JOB_ID:-local}"
RUN="$ECR_HOME/runs/seed${SEED}_${JOB}"
mkdir -p "$RUN/save/redial_rec" "$RUN/log" || exit 1

# Symlink the sources and the read-only data tree.
for p in "$ECR_SRC"/*.py "$ECR_SRC/data"; do
  ln -sfn "$p" "$RUN/$(basename "$p")"
done

# save/ is BOTH an input and an output directory in ECR. The frozen backbones live in
# save/dialogpt and save/roberta (train_rec.py:46-51 default to them), while
# save/redial_rec is where evaluate_rec.py writes its per-run logs. So link the inputs in
# and keep only redial_rec local, or the run cannot find its tokenizer.
for p in "$ECR_SRC"/save/*; do
  b=$(basename "$p")
  case "$b" in redial_rec|train_rec_*|train) continue ;; esac
  ln -sfn "$p" "$RUN/save/$b"
done

for need in save/dialogpt save/roberta data/saved/pre; do
  [ -e "$RUN/$need" ] || { echo "MISSING: $ECR_SRC/$need -- run ecr_fetch.sh (step 5) first"; exit 1; }
done

OUT="$RUN/save/train"

echo "=== node: $(hostname)  job: $JOB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== seed: $SEED   epochs: $EPOCHS"
echo "=== work dir: $RUN"
echo

# Every flag below is from ECR's README, unchanged, so that a seed comparison is a
# comparison of ECR and not of our own recipe.
# --prompt_encoder is the STAGE 1 checkpoint; stage 2 starts from it and adds the
#   recommendation-prefix parameters, which are the five tensors this seed initialises.
# --per_device_eval_batch_size 32 keeps the reported AUC comparable with the paper
#   and with ecr_eval.sh (see auc_batch_size.md).
"$SING" exec --nv --pwd "$RUN" "$ECR_SIF" python "$ECR_BUNDLE/run_ecr.py" train_rec.py \
  --dataset redial_gen \
  --prompt_encoder data/saved/pre/ \
  --output_dir "$OUT" \
  --n_prefix_rec 10 \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 32 \
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
  --seed "$SEED"
STATUS=$?

echo
echo "=== exit status: $STATUS"
# The dump holds one phase per evaluation round, two per epoch (valid then test), in
# order. With 5 epochs that is phases 0-9, and the last two are the final epoch's.
echo "=== AUC against evaluation batch size, final two phases are this run's end state ==="
"$SING" exec --pwd "$RUN" "$ECR_SIF" python "$ECR_BUNDLE/auc_sweep.py" \
  "$RUN/save/redial_rec/auc_dump.jsonl"

echo
# train_rec.py:132 appends a "%Y-%m-%d-%H-%M-%S" stamp to --output_dir with no separator,
# so the real directory is "$OUT<stamp>", holding best/ and final/ (model_prompt.py:284).
echo "=== checkpoints ==="
ls -lh "$OUT"*/best/model.pt "$OUT"*/final/model.pt 2>/dev/null \
  || echo "(no model.pt -- validation loss never improved, or the run stopped early)"
exit $STATUS
