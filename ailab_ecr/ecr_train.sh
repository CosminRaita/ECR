#!/bin/bash
#SBATCH --job-name=ecr-train
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#
# A short train_rec.py run: trains the prompt encoder from the stage-1 checkpoint, then
# runs ECR's usual validation and test passes.
#
#   sbatch ecr_train.sh            # 200 optimiser steps (plumbing check, ~30 min)
#   sbatch ecr_train.sh 2000       # longer run; $1 is the step count
#   sbatch ecr_train.sh 0          # 0 means "no step cap": one full epoch (hours)
#
# Why --mem=64G. The eval-only job peaks at about 6 GB of host RAM. Training adds the
# optimiser state, a retained graph (train_rec.py:421 passes retain_graph=True) and
# TOCoAppear over the 33,868-example training split, which keeps a dense n_entity vector
# per entity. Locally that exceeded a 7.7 GB box and the job started swapping, running 11x
# slower before it was killed. 64G is headroom, not a measured peak -- record the real
# "Maximum resident set size" from this job's output and trim it later if you like.
#
# GPU memory is tighter than evaluation suggests. Eval at batch 32 peaks at 3.0 GiB, but
# TRAINING at batch 16 peaked at 15.30 GiB allocated / 19.07 GiB reserved on an L4 (job
# 915118) -- about 83% of the card. Do not raise the train batch size without checking.
set -uo pipefail
# Slurm runs a COPY of this script from its spool dir, so $0's directory is not the
# bundle. SLURM_SUBMIT_DIR is where you ran sbatch from; $0 covers a direct "bash" run.
BUNDLE_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
[ -f "$BUNDLE_DIR/ecr_env.sh" ] || { echo "cannot find ecr_env.sh next to $BUNDLE_DIR -- submit from the bundle directory, or set ECR_BUNDLE"; exit 1; }
source "$BUNDLE_DIR/ecr_env.sh"
ecr_check_layout || exit 1

STEPS="${1:-200}"
cd "$ECR_SRC"
mkdir -p save/redial_rec log

OUT="save/train_rec_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== output dir: $OUT    step cap: ${STEPS} (0 = full epoch)"
echo

# Warmup must stay below the step count or the learning rate never leaves ~0 and the run
# tells you nothing. ECR's README value (530) suits a multi-thousand-step run.
if [ "$STEPS" = "0" ]; then
  STEP_ARGS="--num_train_epochs 1"
  WARMUP=530
else
  STEP_ARGS="--max_train_steps $STEPS --num_train_epochs 1"
  WARMUP=$(( STEPS / 10 ))
  [ "$WARMUP" -lt 1 ] && WARMUP=1
  [ "$WARMUP" -gt 530 ] && WARMUP=530
fi
echo "=== warmup steps: $WARMUP"
echo

# --prompt_encoder is the STAGE 1 checkpoint: stage 2 starts from it and adds the
#   recommendation-prefix parameters, which are the ones the seed initialises.
# --output_dir is passed explicitly to keep runs apart and job-tagged. It would be safe
#   to omit: train_rec.py:132 appends a timestamp, so the default data/saved/rec becomes
#   data/saved/rec<stamp> and does not touch the authors' released weights.
# --per_device_eval_batch_size 32 keeps the reported AUC comparable with the paper and
#   with ecr_eval.sh (see auc_batch_size.md).
"$SING" exec --nv --pwd "$ECR_SRC" "$ECR_SIF" python "$ECR_BUNDLE/run_ecr.py" train_rec.py \
  --dataset redial_gen \
  --prompt_encoder data/saved/pre/ \
  --output_dir "$OUT" \
  --n_prefix_rec 10 \
  $STEP_ARGS \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --context_max_length 200 \
  --prompt_max_length 200 \
  --entity_max_length 32 \
  --num_warmup_steps "$WARMUP" \
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
echo "=== AUC against evaluation batch size, from this run's scores ==="
"$SING" exec --pwd "$ECR_SRC" "$ECR_SIF" python "$ECR_BUNDLE/auc_sweep.py" \
  save/redial_rec/auc_dump.jsonl

echo
# train_rec.py:132 does `args.output_dir = args.output_dir + local_time`, appending a
# "%Y-%m-%d-%H-%M-%S" stamp with NO separator. So the real directory is "$OUT<stamp>",
# not "$OUT", and it holds best/ and final/ (model_prompt.py:284 writes model.pt).
echo "=== checkpoints (ECR appends a timestamp to --output_dir) ==="
ls -dlh "$OUT"*/ 2>/dev/null || echo "(no output directory found)"
ls -lh "$OUT"*/best/model.pt "$OUT"*/final/model.pt 2>/dev/null   || echo "(no model.pt -- validation loss never improved, or the run stopped early)"
