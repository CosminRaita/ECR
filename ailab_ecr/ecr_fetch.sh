#!/bin/bash
# Downloads ECR's data, its released checkpoints, and the two frozen backbones.
# Network-bound rather than compute-heavy, but it still needs a compute node, because
# "singularity" is NOT on the front-end's PATH -- only the compute nodes have it:
#
#   srun --mem 8G --time=01:00:00 bash ecr_fetch.sh
#
# Compute nodes do have outbound internet (the image build pulls ~2 GB from PyPI). If a
# download is blocked or Google Drive refuses on quota, fetch the two zips on your laptop
# and scp them into ECR/src_emo/downloads/ instead; this script skips what is already
# there, so re-running it then only does the extract and the backbones.
#
# Why the backbones are here: train_rec.py defaults --model/--text_encoder to
# save/dialogpt/ and save/roberta/, which ECR ships EMPTY. Neither Google Drive archive
# contains them. Without this step the first real run dies on a missing model directory.
set -uo pipefail
# Slurm runs a COPY of this script from its spool dir, so $0's directory is not the
# bundle. SLURM_SUBMIT_DIR is where you ran sbatch from; $0 covers a direct "bash" run.
BUNDLE_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
[ -f "$BUNDLE_DIR/ecr_env.sh" ] || { echo "cannot find ecr_env.sh next to $BUNDLE_DIR -- submit from the bundle directory, or set ECR_BUNDLE"; exit 1; }
source "$BUNDLE_DIR/ecr_env.sh"
[ -d "$ECR_SRC" ] || { echo "MISSING: $ECR_SRC (git clone ECR first)"; exit 1; }
[ -f "$ECR_SIF" ] || { echo "MISSING: $ECR_SIF (build the image first)"; exit 1; }

DL="$ECR_SRC/downloads"
mkdir -p "$DL" "$ECR_SRC/data/emo_data" "$ECR_SRC/data/saved" "$ECR_SRC/save" "$ECR_SRC/log"

echo "=== 1/4  data archive (gdown 1fb9kDo8uSRLlwc5c4nUw8DZHR5XOY_l_) ==="
if [ -s "$DL/emo_data.zip" ]; then
  echo "already present, skipping"
else
  "$SING" exec --pwd "$DL" "$ECR_SIF" gdown 1fb9kDo8uSRLlwc5c4nUw8DZHR5XOY_l_ -O emo_data.zip
fi

echo "=== 2/4  released checkpoints (gdown 1uBtcqbQByVrrJ1hEwk2dvsAOxuvEgE19) ==="
if [ -s "$DL/ckpt.zip" ]; then
  echo "already present, skipping"
else
  "$SING" exec --pwd "$DL" "$ECR_SIF" gdown 1uBtcqbQByVrrJ1hEwk2dvsAOxuvEgE19 -O ckpt.zip
fi

echo "=== 3/4  extracting ==="
# Python's zipfile, not unzip: it is in the image for certain, and this skips the
# __MACOSX/.DS_Store entries both archives carry.
"$SING" exec --pwd "$ECR_SRC" "$ECR_SIF" python - <<'PY'
import os, zipfile
for zip_name, dest in [("downloads/emo_data.zip", "data/emo_data"),
                       ("downloads/ckpt.zip", "data/saved")]:
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_name) as z:
        members = [n for n in z.namelist()
                   if not n.startswith("__MACOSX") and not n.endswith(".DS_Store")]
        z.extractall(dest, members=members)
    print(f"  {zip_name} -> {dest}/  ({len(members)} entries)")
PY

echo "=== 4/4  backbones: DialoGPT-small and roberta-base ==="
# PyTorch-format files only: transformers 4.15 predates safetensors, and skipping the
# TF/Flax/ONNX copies saves about 1 GB.
"$SING" exec --pwd "$ECR_SRC" "$ECR_SIF" python - <<'PY'
from huggingface_hub import snapshot_download
ALLOW = ["*.json", "*.txt", "*.bin", "*.model"]
IGNORE = ["*.safetensors", "*.h5", "*.msgpack", "*.onnx", "rust_model.ot"]
for repo, dest in [("microsoft/DialoGPT-small", "save/dialogpt"), ("roberta-base", "save/roberta")]:
    snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=ALLOW, ignore_patterns=IGNORE)
    print(f"  {repo} -> {dest}/")
PY

echo
echo "=== what we now have ==="
echo "-- data/emo_data (conv_unicrs_* are the files merge.py needs):"
ls -1 "$ECR_SRC/data/emo_data" | sed 's/^/     /'
echo "-- data/saved (pre/ = stage 1, rec/ = the released recommender):"
ls -1 "$ECR_SRC/data/saved" | sed 's/^/     /'
echo "-- save:"
ls -1 "$ECR_SRC/save" | sed 's/^/     /'
echo
echo "Sanity check -- all three of these must exist:"
for f in data/emo_data/conv_unicrs_train.jsonl data/emo_data/conv_unicrs_valid.jsonl \
         data/emo_data/conv_unicrs_test.jsonl data/saved/rec/model.pt save/dialogpt/pytorch_model.bin \
         save/roberta/pytorch_model.bin; do
  [ -s "$ECR_SRC/$f" ] && echo "  ok      $f" || echo "  MISSING $f"
done
