# ECR's AUC depends on the evaluation batch size

First measured locally on 2026-10-16/17 (RTX 4060, sm_89, the same GPU generation as the
AI-Lab's L4s), then confirmed on the AI-Lab on 2026-10-20 (job 915107, NVIDIA L4).
Both evaluate ECR's own released recommender checkpoint (`data/saved/rec/`) in `--test`
mode on ReDial, flags as in ECR's README, `--seed 8`. The local run used
`--per_device_eval_batch_size 4`, the cluster run the paper's 32.

## Reproduction of the paper

| metric | paper (ECR row, Table 1) | ours | delta |
|---|---|---|---|
| Recall@1  | 0.049 | 0.0479 | -0.001 |
| Recall@10 | 0.220 | 0.2179 | -0.002 |
| Recall@50 | 0.428 | 0.4283 | +0.000 |
| RT@1      | 0.055 | 0.0544 | -0.001 |
| RT@10     | 0.238 | 0.2341 | -0.004 |
| RT@50     | 0.452 | 0.4510 | -0.001 |
| AUC       | 0.541 | 0.5964 | **+0.055** |

Six of seven match. The AUC outlier is fully explained below, and at the paper's batch size
it matches too (0.5395 vs 0.541).

Cost of one evaluation pass (valid + test): **17.6 min** locally at batch 4 (peak GPU
1.45 GiB allocated / 1.70 reserved, peak host RAM 6.09 GB), **7.7 min** on an L4 at batch 32
(peak GPU 3.01 GiB allocated / 3.96 reserved). Evaluation is cheap on either card.

**Training is not.** At train batch 16 on an L4, job 915118 peaked at **15.30 GiB allocated
/ 19.07 GiB reserved — about 83% of the card's 23 GB**, at 1.15 s/step. Evaluation's memory
figures are no guide to training's.

## The finding

`evaluate_rec.py:55-72` computes AUC **within each batch**, averages the per-batch values
(`auc_batch / batch_count`), and its bare `except: pass` silently drops any batch holding no
liked or no disliked item. So the reported AUC is a function of
`--per_device_eval_batch_size`. Recomputed from one run's logged scores (Patch 3), test split:

| eval batch | AUC | batches kept | batches dropped | % of examples used |
|---|---|---|---|---|
| 2 | 0.5828 | 163 | 2207 | 6.9% |
| 4 | 0.5964 | 191 | 994 | 14.4% |
| 8 | 0.5869 | 167 | 426 | 23.9% |
| 16 | 0.5527 | 139 | 158 | 38.8% |
| 32 (paper's) | 0.5395 | 103 | 46 | 56.8% |
| 64 | 0.5236 | 65 | 10 | 71.2% |
| 128 | 0.5588 | 37 | 1 | 81.4% |
| 512 | 0.5633 | 10 | 0 | 81.4% |
| whole split at once | **0.5486** | 1 | 0 | 81.4% |

**The same weights, data and code score anywhere from 0.524 to 0.596** depending only on how
examples are grouped into batches: a 0.072 spread, against ECR's claimed 0.035 margin over
UniCRS (0.541 vs 0.506). The curve is not monotonic, so there is no "correct direction" to
correct toward.

Two mechanisms:

1. **Discarded data.** ReDial is lopsided -- on the test split, 3601 liked vs 259 disliked
   items, plus 880 with no stated sentiment. AUC needs both classes present in the same
   batch, so at batch 4 the `except: pass` throws away 994 of 1185 batches and only 14.4% of
   examples ever reach the metric. The ceiling is 81.4% even in one giant batch, because
   "not stated" examples can never contribute.
2. **Unweighted averaging.** A batch contributing one liked/disliked pair counts as much as
   one contributing hundreds, so sparse batches dominate the mean.

The defensible number is the last row: **0.5486**, computed once over the whole split, no
batch discarded, every usable example weighted equally.

## Reproducing this

Patch 3 (see `../PATCHES.md`) makes `evaluate_rec.py` write `save/redial_rec/auc_dump.jsonl`:
one line per example with the softmax score given to the gold item, the sentiment label
(1 liked, 0 disliked, -1 not stated), and the batch it fell in. It changes no result. Then:

    python auc_sweep.py save/redial_rec/auc_dump.jsonl

Phase 0 is the valid split, phase 1 is test. The script checks itself against ECR's literal
double loop before reporting (agreed to 0.00e+00 locally). Because the valid and test
dataloaders are built without `shuffle` (`train_rec.py:225-234`), example order is the file
order and offline re-grouping is faithful to a real run at that batch size.

## The confirmation run, and the limit of the offline sweep

The sweep assumes a per-example score is independent of the batch it sits in. It is not,
quite: sequences are padded to the longest member of their batch, so grouping perturbs the
scores slightly. The cluster run at batch 32 lets us put a number on that.

| | offline sweep at batch 32, from the batch-4 run | real run at batch 32 (job 915107) |
|---|---|---|
| AUC | 0.5395 | 0.54064 |
| Recall@10 | — | 0.22025 (vs 0.2179 measured at batch 4) |

Two readings of the same table. The AUC prediction lands within **0.001** of the real run,
against the 0.072 effect being measured — so the sweep is sound for the finding. But
Recall@10, which involves no batch-averaging at all and should have been invariant, still
moved **+0.0024** between batch 4 and batch 32. That is the padding effect, visible in a
metric where nothing else could have caused it.

So: **the offline sweep is a very good approximation, not an identity.** Quote the curve for
the shape of the effect and for the whole-split number, and quote a real run for any single
figure that goes in a table. Both runs' dumps should be kept; the job scripts copy
`auc_dump.jsonl` to a job-tagged name, because Patch 3 reopens the fixed path in `w` mode
and the next job would otherwise truncate it.

## Why this matters for SeedGuard

This is an evaluation-protocol fragility found before a single seed was changed: a headline
number that moves more with a batch-size flag than with the modelling contribution it is
meant to demonstrate. It is the same class of problem as seed sensitivity -- a result that
looks like a property of the model but is partly a property of the harness -- and it argues
for SeedGuard recording the evaluation configuration alongside the metric, not just the
metric.
