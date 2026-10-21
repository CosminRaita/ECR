#!/bin/bash
#SBATCH --job-name=ecr-prepare
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#
# ECR's data preparation, up to exactly what train_rec.py --dataset redial_gen needs.
# CPU only, no GPU requested. Run AFTER ecr_fetch.sh:
#
#   sbatch ecr_prepare.sh
#
# Three deliberate deviations from ECR's README, each found by running it locally:
#
#  1. Each script runs with the working directory set to its OWN data directory. They open
#     bare filenames ("entity2id.json"), so the README's "python data/redial/process.py"
#     from src_emo looks in the wrong place and crashes.
#  2. process.py runs BEFORE process_mask.py. Only process.py writes item_ids.json, which
#     the DBpedia loader needs; process_mask.py then overwrites the processed files with
#     the <movie>-masked versions the recommender stage expects.
#  3. movie_reviews_filted_0.1_confi.json (592 MB) is not copied into data/redial/. It is
#     read only by the emotional-reply stage, which is out of scope for T1.
#
# data/redial/ stays populated on purpose: evaluate_rec.py and co_appear.py hard-code
# "data/redial/..." even when --dataset is redial_gen.
set -euo pipefail
# Slurm runs a COPY of this script from its spool dir, so $0's directory is not the
# bundle. SLURM_SUBMIT_DIR is where you ran sbatch from; $0 covers a direct "bash" run.
BUNDLE_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
[ -f "$BUNDLE_DIR/ecr_env.sh" ] || { echo "cannot find ecr_env.sh next to $BUNDLE_DIR -- submit from the bundle directory, or set ECR_BUNDLE"; exit 1; }
source "$BUNDLE_DIR/ecr_env.sh"
ecr_check_layout || exit 1

cd "$ECR_SRC"
mkdir -p data/redial data/redial_gen save/redial_rec log

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo

echo "=== copy emo_data -> data/redial/ (excluding the 592 MB review file) ==="
for f in data/emo_data/*; do
  base=$(basename "$f")
  [ "$base" = "movie_reviews_filted_0.1_confi.json" ] && continue
  cp -n "$f" "data/redial/$base" 2>/dev/null || true
done
ls -1 data/redial | sed 's/^/     /'
echo

echo "=== process.py  (writes *_data_processed.jsonl and item_ids.json) ==="
time "$SING" exec --pwd "$ECR_SRC/data/redial" "$ECR_SIF" python "$ECR_SRC/data/redial/process.py"
echo

echo "=== process_mask.py  (overwrites the processed files, masked; writes movie_ids.json) ==="
time "$SING" exec --pwd "$ECR_SRC/data/redial" "$ECR_SIF" python "$ECR_SRC/data/redial/process_mask.py"
echo

echo "=== copy data/redial/* -> data/redial_gen/ ==="
cp -r data/redial/* data/redial_gen/
echo

echo "=== merge.py  (splices UniCRS conversation output into the 'resp' field) ==="
time "$SING" exec --pwd "$ECR_SRC/data/redial_gen" "$ECR_SIF" python "$ECR_SRC/data/redial_gen/merge.py"
echo

echo "=== does the prepared data match the released checkpoint? ==="
"$SING" exec --pwd "$ECR_SRC" "$ECR_SIF" python - <<'PY'
import json, torch
D = "data/redial_gen"
entity2id = json.load(open(f"{D}/entity2id.json"))
item_ids = json.load(open(f"{D}/item_ids.json"))
relation2id = json.load(open(f"{D}/relation2id.json"))
num_entities = max(entity2id.values()) + 2          # dataset_dbpedia.py:60
sd = torch.load("data/saved/rec/model.pt", map_location="cpu")
ckpt_entities, hidden = sd["node_embeds"].shape
ckpt_relations = sd["kg_encoder.comp"].shape[0]
print(f"  entities   data={num_entities:6d}  checkpoint={ckpt_entities:6d}  "
      f"{'ok' if num_entities == ckpt_entities else 'MISMATCH'}")
print(f"  relations  data={len(relation2id):6d}  checkpoint={ckpt_relations:6d}  "
      f"{'ok' if len(relation2id) == ckpt_relations else 'MISMATCH'}")
print(f"  rankable items: {len(item_ids)}   hidden size: {hidden}")
for split in ("train", "valid", "test"):
    n = rec = resp = 0
    for line in open(f"{D}/{split}_data_processed.jsonl", encoding="utf-8"):
        d = json.loads(line); n += 1
        rec += bool(d.get("rec")); resp += bool(d.get("resp"))
    print(f"  {split:5s} {n:6d} turns | {rec:6d} with a recommendation | {resp:6d} with a merged response")
PY
echo
echo "Prepared. Next: sbatch ecr_eval.sh"
