# Plan: T0 — ECR code tour, plus the blocker list T1 will need

## Context

I read `SeedGuard/CLAUDE.md` in full and went through ECR's recommender path in `ECR/src_emo/`
(`train_pre.py`, `train_rec.py`, `config.py`, `model_prompt.py`, `model_gpt2.py`, `dataset_rec.py`,
`dataset_dbpedia.py`, `co_appear.py`, `evaluate_rec.py`, the `data/redial*/process*.py` + `merge.py`
preprocessing), ECR's README, the `ailab_ecr/` bundle (now in `ECR/ailab_ecr/`), and
`SeedGuard/documentation/architecture.md`.

CLAUDE.md §9 puts **T0 (ECR code tour, read-only)** first, and T1 (ECR on the AI-Lab) as the main
goal this week. T0's deliverable is `SeedGuard/documentation/ecr_code_tour.md`. While reading I also
found several concrete things that will break a cluster run and are **not** in CLAUDE.md §5.6 or in
`ailab_ecr/README.md`. Those belong in the tour document (and later in `PATCHES.md`) so T1 doesn't
rediscover them one crash at a time.

Current state of the workspace, for the record:
- `SeedGuard/` = `CLAUDE.md`, `README.md` (2 lines), `documentation/architecture.md` + diagram. No code yet.
- `ECR/` = upstream clone, untouched (git log is only README commits). No data, no checkpoints:
  `src_emo/data/emo_data/` and `src_emo/data/saved/` hold only `note.md` ("unzip … here"), and
  `src_emo/save/` (expected to hold `dialogpt/` + `roberta/`) is empty.
- `ECR/ailab_ecr/` = `ecr.def`, `gpu_check.py`, `gpu_check.sh`, `ecr_smoke.py`, `README.md`.

## Verified understanding (what goes into the tour)

**Three stages.** `train_pre.py` (semantic fusion, `--seed 42`) → `train_rec.py` (recommendation,
`--seed 8`, the MVP stage) → `train_emp.py`/`infer_emp.py` (emotional replies, out of scope).

**What trains in stage 2** — confirms CLAUDE.md §5.3:
- `train_rec.py:165-167` freezes `model` (DialoGPT, `PromptGPT2forCRS`) and `text_encoder` (RoBERTa);
  HF `from_pretrained` leaves them in eval mode and they are never `.train()`ed → no dropout.
- `train_rec.py:170` `modules = [prompt_encoder]` is the only thing in the optimiser. `KGPrompt`
  contains no dropout layers at all, so the trainable part has **no stochastic forward pass**.
- `prompt_encoder.load()` (`model_prompt.py:290`) loads stage 1 with `strict=False`. Stage 1 builds
  `KGPrompt` *without* `n_prefix_rec` (`train_pre.py:155`), stage 2 *with* it (`train_rec.py:154`),
  so `rec_prefix_embeds` + `rec_prefix_proj` are new and seed-initialised (`model_prompt.py:87-93`).
- Seed therefore controls: new-parameter init, `DataLoader(shuffle=True)` order
  (`train_rec.py:217`), the `random_split` call at `train_rec.py:194` (no-op at `--shot 1`), and GPU
  non-determinism. Nothing else.

**Determinism at inference** — confirms CLAUDE.md §4.4: `model_gpt2.py:717-718` builds
`rec_logits = last_token_hidden @ entity_embeds.T`, `train_rec.py:467` restricts to `kg['item_ids']`,
`torch.topk(..., k=50)`. One forward pass, full catalogue, no sampling, no evaluation seed.

**Metrics.** `evaluate_rec.py` already computes Recall/MRR/NDCG@{1,10,50}, plus `recall_true@k`
(RT@k — recall over liked targets only), per-batch AUC (`evaluate_AUC`, liked vs disliked pairs) and
an MSE. `train_rec.py` prints valid **and test** reports every epoch, so one short run yields the
numbers T1 asks for. Note `--test` mode (`train_rec.py:269`) evaluates a given checkpoint without
training — that is the hook a SeedGuard adapter's `evaluate()` will reuse.

**Gap for SeedGuard.** Nothing writes ranked lists: `evaluate_rec.py:36-44` logs only the **top-1**
item name, and only in `--test` mode. Section 8 correction 1 (adapters return ranked lists so
metrics and Jaccard/RBO are computed later from stored predictions) needs a dump of the top-50
`ranks` + label per example. That is an ECR patch to propose when we reach T2/T3 — not now.

## Blockers found that T1 will hit (not in CLAUDE.md §5.6)

| # | Where | Problem | Fix (outside ECR where possible) |
|---|---|---|---|
| A | `train_rec.py:6` | `os.environ["CUDA_VISIBLE_DEVICES"] = "3"` runs before torch import. Under `sbatch --gres=gpu:1` Slurm exposes exactly one GPU as device 0, so torch sees **no** GPU and `dataset_dbpedia.py:55` `.cuda()` crashes. | Patch (delete/comment the line, as `train_emp.py:6` already is). No env-var workaround: the line overwrites whatever Slurm sets. |
| B | `evaluate_rec.py:17` | `open("save/redial_rec/rec.json", 'w')` — directory does not exist → `FileNotFoundError` at evaluator construction, before any training. | `mkdir -p src_emo/save/redial_rec` in the sbatch script. No ECR edit. |
| C | `evaluate_rec.py:20`, `co_appear.py:59` | Hardcode `data/redial/entity2id.json` and `data/redial/train_data_dbpedia_emo.jsonl` even when `--dataset redial_gen`. | None needed if we follow the README's `cp -r data/redial/* data/redial_gen/`, i.e. keep `data/redial/` populated. Document it. |
| D | `train_rec.py:44-49` | `--tokenizer/--model/--text_tokenizer/--text_encoder` default to `save/dialogpt/` and `save/roberta/`, which are **empty**. Cluster nodes may be offline, so HF auto-download can't be assumed. | Pre-fetch `microsoft/DialoGPT-small` + `roberta-base` into `src_emo/save/` on the front-end (or pass HF ids if nodes have internet). Check first whether the Drive ckpt zip ships them. |
| E | `train_rec.py:52` vs `train_pre.py:29` | `--prompt_encoder` defaults to `data/saved/pre/`, but stage 1 writes `data/saved/pre-trained/`. | Pass `--prompt_encoder` explicitly. Confirm which name the Drive ckpt bundle uses. |
| F | `data/redial/process.py`, `process_mask.py`, `data/redial_gen/merge.py` | All open **cwd-relative** filenames (`entity2id.json`, `train_data_processed.jsonl`). ECR's README says `python data/redial/process.py` from `src_emo/`, which looks in the wrong directory. | `cd` into each data dir before running, in our pipeline script. No ECR edit. |
| G | `merge.py:14` | Needs `conv_unicrs_{train,valid,test}.jsonl` (UniCRS conversation-model output). Unverified that the Drive bundle contains them; if not, stage 2's `redial_gen` input cannot be built. | Check the Drive zip contents first (T1 step 6). Flag early — this is the one item that could block T1 outright. |
| H | `co_appear.py:145-153` + `dataset_rec.py:121` + `model_prompt.py:162-171` | `TOCoAppear` keeps a dense `n_entity`-float vector per entity (ReDial ≈ 30k entities → ~3.5 GB CPU RAM), the collator builds `(batch, entity_len, n_entity)` tensors, and `get_nei_rep` (used because `--nei_mer` is on) materialises `(batch, n_entity, hidden)` ≈ 1.5 GB at batch 16. | Not a bug — explains ECR's "24 GB GPU". Request generous `--mem` (≥48G) in sbatch and keep `--per_device_train_batch_size 16`; be ready to halve it. Record actual peak in T1 step 7. |
| I | `config.py:19-32` `seed_torch` | Sets `CUDA_LAUNCH_BLOCKING=1` and `cudnn.enabled=False` (both slow), and sets `CUBLAS_WORKSPACE_CONFIG` *after* torch is imported, so it may come too late. With patch 1's `warn_only=True`, RGCN's scatter has no deterministic kernel → expect warnings and possible same-seed drift on GPU. | Don't "fix". **Measure** it: this is exactly the hidden randomness SeedGuard's determinism check exists to flag. |

Patch 1 (`torch.set_deterministic` → `torch.use_deterministic_algorithms(..., warn_only=True)`) and
patch 2 (the `torch_sparse`-free path) stay as CLAUDE.md §5.6 describes. Note `ecr_smoke.py:14`
monkey-patches `torch.set_deterministic`, so the smoke test runs **without** editing ECR at all —
patch 1 only becomes necessary for the real `train_rec.py` run (bundle README step 5).

## Gaps in the `ailab_ecr/` bundle

- `ecr.def:13` installs `torch_scatter`/`torch_sparse` from the **torch-2.4.0+cu121** wheel index
  while installing torch **2.4.1**. Usually fine, but if no matching cp39 wheel is found pip falls
  back to a source build that needs a CUDA toolkit and will fail or hang. This is the most likely
  build failure; the fallback is already documented (drop them → patch 2).
- `gpu_check.py:18` imports `torch_sparse` unconditionally, so the script ends in an `ImportError`
  when it is missing instead of reporting cleanly. Cosmetic, worth a one-line guard.
- `gpu_check.py:13` `matmul ok` compares a matmul against itself — it proves kernels launch, not
  that results are right. Fine as a smoke test; say so rather than overclaiming.
- No sbatch script for the smoke test (bundle README step 4 uses `srun`, which ties up your session)
  and **none for `train_rec.py`** — that is T1 step 7's deliverable.
- No `PATCHES.md` yet.

## Deliverable for this step (T0)

Write `SeedGuard/documentation/ecr_code_tour.md` — plain language, no jargon unintroduced, covering:
1. Map of `src_emo/`: which file is which stage, and the three-layer DialoGPT / UniCRS / ECR split.
2. Where text and graph mix: `KGPrompt.forward` (`model_prompt.py:174`) — RGCN entity embeddings
   (`get_entity_embeds`, line 130), RoBERTa token embeddings, `cross_attn`, the emotion fusion, and
   the `nei_mer` branch; output is per-layer prefix key/values for the frozen GPT-2.
3. What is frozen vs trained, with file:line evidence, and the "what the seed controls" list above.
4. Where a recommendation comes from (`model_gpt2.py:717` → `topk`) and why it is deterministic.
5. Where metrics live (`evaluate_rec.py`), which ones ECR reports, and the ranked-list gap.
6. Where seeds are set (`config.py:seed_torch`, `train_pre.py:91-95`'s `--use_new_seed` branch,
   `train_rec.py:96`) and which RNG each thing draws from.
7. The blocker table above, as the input to `PATCHES.md`.

Acceptance (CLAUDE.md §9): you can explain the three stages and what the seed controls from the notes.

## Then, on a separate "go" (T1 prep, no cluster commands from me)

1. Add to the bundle: `ecr_smoke.sh` and `train_rec_short.sh` sbatch scripts (the latter doing
   `mkdir -p save/redial_rec`, pinning `--prompt_encoder`, 1 epoch / small `--max_train_steps`,
   printing Recall@1/10/50 + MRR + NDCG, and reporting wall time and peak memory).
2. Write `PATCHES.md` and apply patch 1 + blocker A only after your explicit approval (§2.2).
3. A `prepare_data.sh` that does the pipeline in the right working directories (blocker F) and
   checks the Drive zips for `conv_unicrs_*.jsonl`, `save/dialogpt`, `save/roberta`, `data/saved/pre`
   (blockers D, E, G) before anything is launched.

I prepare and explain; you run everything on the cluster.

## Verification

- T0 is documentation only — nothing to execute. Check by reading, against CLAUDE.md §5.3/§5.4.
- Every claim in the tour cites `file:line`, so each is checkable in `ECR/src_emo/` without a GPU.
- The blocker list gets verified for real at T1: blockers A, B, D, E, F each produce a crash on the
  first real `train_rec.py` run if we get them wrong, so the short run is the test.
- No files are created in `SeedGuard/` or edits made to `ECR/` until you say go (§2.2, §9 T0).
