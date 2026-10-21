# Patches applied to ECR

ECR (https://github.com/zxd-octopus/ECR) is third-party code and is treated as read-mostly.
Every change we make to it is listed here, with the reason and the exact edit, so the
working copy can be reconstructed from a fresh clone.

Base clone: `26e2342` ("Update README.md").

---

## Patch 1 — `src_emo/config.py`: renamed determinism call

**Why.** `torch.set_deterministic()` was removed from PyTorch after 1.8. Our stack is
torch 2.4.1 (ECR's pinned 1.8.1 + CUDA 11.1 predates the Ada GPUs we run on: RTX 4060
locally, L4 on the AI-Lab, both sm_89). Without this, `seed_torch()` raises
`AttributeError` on every run.

`warn_only=True` because some operations ECR uses — notably the RGCN scatter in the graph
encoder — have no deterministic GPU implementation. Erroring out would make the run
impossible; warning tells us where non-determinism remains, which is itself a finding for
SeedGuard.

```diff
     print(f"Random seed set as {seed}")
-    torch.set_deterministic(True)
+    torch.use_deterministic_algorithms(True, warn_only=True)
```

Agreed in advance: `SeedGuard/CLAUDE.md` §5.6 item 1.

---

## Patch 2 — `src_emo/train_rec.py`: hard-coded GPU index

**Why.** Line 6 pinned the process to GPU 3 before torch is imported:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
```

On any machine with fewer than four GPUs this leaves torch with **no** visible device, and
`dataset_dbpedia.py:55`'s `.cuda()` then fails. That covers both of our targets: a
single-GPU laptop, and an AI-Lab job requesting `--gres=gpu:1`, where Slurm exposes the
allocated GPU as device 0 regardless of its physical index. The line also overrides
whatever Slurm sets, so it cannot be worked around from outside the code.

Commented out rather than deleted, to keep the upstream line visible:

```diff
-os.environ["CUDA_VISIBLE_DEVICES"] = "3"
+# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
```

`train_emp.py:6` already has the same line commented out upstream, so this matches the
authors' own treatment of it.

Not listed in CLAUDE.md §5.6; approved explicitly on 2026-10-16.

---

## Patch 3 — `src_emo/evaluate_rec.py`: per-example AUC logging

**Why.** ECR's `evaluate_AUC` computes AUC **within each batch** and averages the per-batch
values (`auc_batch / batch_count`), and its bare `except: pass` silently drops any batch
that contains no liked or no disliked item. The reported AUC therefore depends on
`--per_device_eval_batch_size`: our first reproduction run at batch 4 gave 0.596 against the
paper's 0.541 at batch 32, while every recall metric matched to within 0.004.

Rather than re-run the model once per batch size, this logs the two values `evaluate_AUC`
consumes — the softmax score the model gave the **gold** item, and that example's sentiment
label (1 liked, 0 disliked, −1 not stated) — one line per example. The test split is not
shuffled, so AUC at any batch size, and a single global AUC over the whole split, can be
recomputed from that one file in seconds on the CPU.

**This changes no result.** The dump is written before the AUC sum, from the function's own
arguments; no metric, logit, ranking or loss is touched, and nothing is read back.

```diff
 # __init__
         self.log_cnt = 0
+        self.auc_dump = open("save/redial_rec/auc_dump.jsonl", 'w', buffering=1)
+        self.auc_phase = 0
+        self.auc_batch_idx = 0

 # evaluate_AUC, after `pre` is built
+        for i in range(len(pre)):
+            self.auc_dump.write(json.dumps({
+                'phase': self.auc_phase, 'batch': self.auc_batch_idx,
+                'score': pre[i], 'sent': float(mse_met[0][i]),
+                'label': int(labels.tolist()[i]),
+            }) + '\n')
+        self.auc_batch_idx += 1

 # reset_metric
+        if hasattr(self, 'auc_phase'):
+            self.auc_phase += 1
+            self.auc_batch_idx = 0
```

`train_rec.py` calls `reset_metric()` between the valid and test loops, so `phase` separates
them: in `--test` mode phase 0 is valid and phase 1 is test. The `hasattr` guard exists
because `__init__` calls `reset_metric()` before these attributes are set.

Output: `save/redial_rec/auc_dump.jsonl`.

Not listed in CLAUDE.md §5.6; approved explicitly on 2026-10-16.

---

## Considered and *not* patched

**`data/saved/pre-trained` vs `data/saved/pre`.** `train_pre.py` defaults to
`--output_dir data/saved/pre-trained`, while `train_rec.py` defaults to
`--prompt_encoder data/saved/pre/`. The authors' checkpoint archive already unpacks to
`data/saved/pre/` and `data/saved/rec/`, so loading their weights needs no change. If we
ever run stage 1 ourselves, pass `--output_dir data/saved/pre` (or point
`--prompt_encoder` at `pre-trained`) on the command line instead of editing the defaults.

**`evaluate_rec.py:17` writes to `save/redial_rec/rec.json`** without creating the folder,
so the evaluator crashes at construction if it is missing. Handled by `mkdir -p` in our run
scripts, not by editing ECR.

**`evaluate_rec.py:20` and `co_appear.py:59` hard-code `data/redial/...`** even when
`--dataset redial_gen` is given. Harmless as long as we follow ECR's README and keep
`data/redial/` populated (`cp -r data/emo_data/* data/redial/`). Documented, not patched.

**Preprocessing scripts use paths relative to the current directory.**
`data/redial/process.py`, `process_mask.py` and `data/redial_gen/merge.py` open bare
filenames, so ECR's README command `python data/redial/process.py` looks in the wrong
place. Our run scripts `cd` into each data directory first. No code change.

**Known upstream quirks left alone** (CLAUDE.md §5.6): `dataset_dbpedia.py` hard-codes
`.cuda()`; `seed_torch` disables cuDNN and sets `CUDA_LAUNCH_BLOCKING=1` (slow but
deterministic); wandb is imported unconditionally — we run with `WANDB_MODE=offline`.

---

## Reserved

**Patch 4 (conditional) — no `torch_sparse`.** Not needed: `torch_sparse 0.6.18+pt24cu121`
installed and ran fine on torch 2.4.1. If a future environment lacks it, swap
`model_prompt.py` line 132 for the commented line 133 and stop wrapping `edge_index` in a
`SparseTensor` in `dataset_dbpedia.py` lines 55-57 (CLAUDE.md §5.6 item 2).

**Patch 5 (likely, for SeedGuard) — dump ranked lists.** `evaluate_rec.py` currently logs
only the top-1 item, and only in `--test` mode. SeedGuard needs the full top-50 ranking per
example so metrics and agreement measures (Jaccard, RBO) can be computed later from stored
predictions without re-running on a GPU. To be proposed when we reach T2/T3.
