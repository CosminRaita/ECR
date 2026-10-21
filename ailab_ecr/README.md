# Running ECR on the AAU AI-Lab

AI-Lab compute nodes have NVIDIA L4 GPUs (24 GB, compute capability sm_89). ECR pins
PyTorch 1.8.1 + CUDA 11.1, which predates the L4 and cannot run on it. Juan's `gpu_test.py`
can still print "CUDA is available", because it never runs anything on the GPU.

`ecr.def` keeps ECR's pinned `transformers==4.15.0` and `accelerate==0.8.0` (ECR imports
internals of that transformers version) but moves to PyTorch 2.4.1 + CUDA 12.1, which
supports the L4. That swap needs three small patches to ECR, all recorded in
`../PATCHES.md`.

**This whole pipeline has been run end to end, first on an RTX 4060** (8 GB, sm_89 — the
same GPU generation as the L4) **and then on the AI-Lab itself** (jobs 914259, 915107,
915118). The reproduction matched the paper on all seven metrics on both. So the scripts
here are not a first guess: the expected numbers, the memory figures and the three
deviations from ECR's README are all measured. See `auc_batch_size.md`.

## What is in this folder

| file | what it is |
|---|---|
| `ecr.def` | Singularity image definition |
| `ecr_env.sh` | shared paths, sourced by the scripts (not run directly) |
| `gpu_check.py`, `gpu_check.sh` | GPU test that runs real ops, including ECR's RGCN layer |
| `ecr_smoke.py` | one training step with ECR's own model code, no data; checks seed reproducibility |
| `ecr_smoke.sh` | **sbatch**: `gpu_check.py` + `ecr_smoke.py` in one job |
| `ecr_fetch.sh` | **srun**: downloads data, checkpoints and the two frozen backbones |
| `ecr_prepare.sh` | **sbatch**, CPU only: ECR's data pipeline up to what `train_rec.py` needs |
| `ecr_eval.sh` | **sbatch**, GPU: evaluates ECR's released checkpoint (the reproduction check) |
| `ecr_train.sh` | **sbatch**, GPU, 64G: a short `train_rec.py` run (plumbing check) |
| `ecr_seed.sh` | **sbatch**, GPU, 48G: one real multi-epoch run at a given seed; several can run at once |
| `run_ecr.py` | launcher that adds wall time and peak GPU memory to any ECR script |
| `auc_sweep.py` | recomputes AUC at any batch size from an eval run's logged scores |
| `auc_batch_size.md` | the reproduction table, measured costs, and the AUC finding |
| `t0_plan_notes.md` | the code-tour notes and the full list of blockers found by reading ECR |

## Layout the scripts expect

Put the bundle and the clone side by side, and **submit jobs from the bundle directory**
(the scripts find each other through `SLURM_SUBMIT_DIR`):

```
~/ecr/                  <- cd here, sbatch from here
  ecr.sif               <- built in step 2
  *.sh *.py             <- this bundle
  ECR/                  <- git clone; ECR/src_emo is where the work happens
```

Override with `ECR_HOME`, `ECR_SIF`, `ECR_SRC` if you prefer a different layout.

## Steps

Everything goes through Slurm. `singularity` is not available on the front-end at all,
so the only things you run there are `git`, `scp` and the `sbatch`/`srun` commands themselves.

```bash
# 1. get the code and this bundle onto the front-end
mkdir -p ~/ecr && cd ~/ecr
git clone https://github.com/zxd-octopus/ECR.git
#    copy this folder's contents into ~/ecr (WinSCP, or scp -r ailab_ecr/* user@host:~/ecr/)

# 2. build the image -- stay connected until it finishes (~10-20 min)
#    The build sandbox goes in TMPDIR, and the node's /tmp is too small for it:
#    without this you get "No space left on device" partway through %post.
mkdir -p ~/sing_tmp
export SINGULARITY_TMPDIR=~/sing_tmp
export APPTAINER_TMPDIR=~/sing_tmp
srun --mem 40G --time=02:00:00 singularity build --fakeroot ecr.sif ecr.def

# 3. check the GPU and ECR's model code (~2 min)
sbatch ecr_smoke.sh
squeue --me                     # wait for it to leave the queue
cat slurm-<ID>.out

# 4. apply the three patches from ../PATCHES.md to the clone
#    (patch 1: config.py determinism call; patch 2: the hard-coded GPU index in
#     train_rec.py; patch 3: the AUC logging in evaluate_rec.py)

# 5. download data, checkpoints and backbones (~800 MB)
#    Needs a compute node: "singularity" is not on the front-end's PATH.
srun --mem 8G --time=01:00:00 bash ecr_fetch.sh

# 6. prepare the data (CPU job, ~10 min)
sbatch ecr_prepare.sh
cat slurm-<ID>.out

# 7. the reproduction check: evaluate ECR's released checkpoint (~20-30 min)
sbatch ecr_eval.sh
cat slurm-<ID>.out

# 8. a short training run (~30 min at the default 200 steps)
sbatch ecr_train.sh
sbatch ecr_train.sh 2000        # longer; $1 is the step count, 0 means a full epoch
```

Monitor with `squeue --me`, follow with `tail -f slurm-<ID>.out`, cancel with `scancel <ID>`.

**Steps 5-8 must not overlap.** Slurm starts jobs as soon as nodes are free, so submitting
them together runs them in parallel and the later ones read data the earlier ones have not
written yet. Either wait for each, or chain them:

```bash
A=$(sbatch --parsable ecr_prepare.sh)
B=$(sbatch --parsable --dependency=afterok:$A ecr_eval.sh)
sbatch --dependency=afterok:$B ecr_train.sh
```

`afterok` means "only if that job exited 0"; a failure leaves the rest queued with reason
`DependencyNeverSatisfied`, which you clear with `scancel`.

Note: one line of Juan's email says `singularity --nv exec`. The correct order is
`singularity exec --nv`, which is what these scripts use.

## What to expect

**Step 3** — `CUDA available: True`, device name `NVIDIA L4`, `matmul ok: True`,
`RGCN ... ok`, `torch_sparse ... ok`, then
`same seed reproducible: True | different seed differs: True`.

If `sm_89` is missing from the compiled architecture list, that alone is **not** a failure:
Ada cards run sm_86 cubins. The `matmul ok` and `RGCN ok` lines are the real evidence.

**Step 5** — the sanity block at the end must show `ok` for all six files. The three
`conv_unicrs_*.jsonl` files are the ones `merge.py` needs; without them the `redial_gen`
dataset cannot be built at all.

**Step 7** — the paper's ECR row, and what we measured locally:

| metric | paper | AI-Lab L4, eval batch 32 (job 915107) | local RTX 4060, eval batch 4 |
|---|---|---|---|
| recall@1 | 0.049 | 0.04873 | 0.0479 |
| recall@10 | 0.220 | **0.22025** | 0.2179 |
| recall@50 | 0.428 | 0.42764 | 0.4283 |
| recall_true@1 | 0.055 | 0.05526 | 0.0544 |
| recall_true@10 | 0.238 | **0.23827** | 0.2341 |
| recall_true@50 | 0.452 | 0.45154 | 0.4510 |
| auc | 0.541 | **0.54064** | 0.5395 |

The L4 column is the one to quote: it is at the paper's own evaluation batch size.
Cost there: 7.7 min, peak GPU 3.01 GiB allocated / 3.96 GiB reserved, on a 23 GB card.

**AUC is reported per batch and averaged, so it moves with
`--per_device_eval_batch_size`** — from 0.524 to 0.596 on the same weights. Both job
scripts therefore pin eval batch 32, the paper's value, and print the whole curve at the
end via `auc_sweep.py`. Read `auc_batch_size.md` before quoting an AUC anywhere.

**Step 8** — a loss per epoch, then the same validation/test report. At 200 steps this is a
plumbing check, not a trained model: it proves the backward pass, optimiser step and
checkpoint save work. Measured on the AI-Lab (job 915118, 200 steps at train batch 16):
1.15 s/step, train loss 12.19, test recall@10 0.162 — low, as expected for 200 steps —
and **peak GPU 15.30 GiB allocated / 19.07 GiB reserved, about 83% of the L4's 23 GB.**

That last figure is the important one. Evaluation's 3 GiB is no guide to training: at batch
16 there are only ~4 GiB spare, so train batch size cannot be raised to buy speed. Host
RAM is still unmeasured — run `seff <ID>` after a job and trim `--mem` from that; `64G`
here is headroom, not a measurement.

## Two traps worth knowing

- **`--output_dir` gets a timestamp appended** (`train_rec.py:132`:
  `args.output_dir = args.output_dir + local_time`, no separator). So passing
  `--output_dir save/train_rec_123` really writes `save/train_rec_1232026-10-20-11-53-18`,
  holding `best/model.pt` and `final/model.pt`. Two consequences: the default
  `data/saved/rec` becomes `data/saved/rec<stamp>` and does **not** overwrite ECR's
  released checkpoint, and any script looking for the directory it asked for will not
  find it. `ecr_train.sh` globs for the stamped name.
- **`evaluate_rec.py` opens `save/redial_rec/rec.json` without creating the folder**, so a
  missing directory crashes the evaluator before any work happens. The scripts `mkdir -p`
  it. Same for the AUC dump, which is reopened in write mode every run — the scripts keep a
  job-tagged copy, because otherwise the next job silently truncates it.

## If something fails

- **`torch_sparse` won't install**: drop it from `ecr.def`. ECR then needs two more edits
  (patch 4 in `../PATCHES.md`): in `model_prompt.py` swap line 132 for the commented line
  133, and in `dataset_dbpedia.py` stop wrapping `edge_index` in a `SparseTensor`
  (lines 55-57). `ecr_smoke.py` tests that path automatically when `torch_sparse` is absent.
- **"no kernel image is available"**: the PyTorch build does not support the GPU. Check what
  `gpu_check.py` prints for the compiled arch list.
- **`FATAL: Couldn't determine user account information: user: unknown userid NNNNN`**: a
  node fault, not yours. That node's directory service is not resolving your UID, so
  Singularity cannot map you into the container and exits in about a second with code 127.
  Nothing in the job ever runs. Resubmit avoiding it:

      sbatch --exclude=ailab-l4-03 ecr_seed.sh 1

  Seen on `ailab-l4-03` on 2026-10-21, while `ailab-l4-04` was healthy at the same moment.
  Worth reporting to AI-Lab support: it breaks every container job scheduled onto that node.
  `sacct -j <ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList%14` is the quickest
  way to tell this apart from a real failure -- an `Elapsed` of one second means the job
  never started, so look at the node, not the code.
- **A preprocessing script cannot find a file**: it is being run from the wrong directory.
  ECR's scripts open bare filenames; `ecr_prepare.sh` runs each one with the working
  directory set to its own data folder, which is why it deviates from ECR's README.
- **Warnings about non-deterministic operations**: expected on GPU, and `warn_only=True` is
  deliberate. Same-seed runs may then differ slightly. Report it, don't hide it — it is a
  source of randomness SeedGuard exists to flag. Locally, zero such warnings fired during
  evaluation.

## What the seed controls in ECR's recommender stage

In `train_rec.py`, DialoGPT and the RoBERTa text encoder are frozen and stay in eval mode
(no dropout). Only the prompt encoder trains, and it starts from the stage-1 checkpoint.
So the recommender-stage seed changes only:

- the initial values of the new recommendation-prefix parameters,
- the order of the training data (shuffled each epoch),
- GPU non-determinism, if any.

Confirmed locally by diffing the two released checkpoints: `rec/model.pt` has exactly five
keys that `pre/model.pt` lacks (`rec_prefix_embeds` and the four `rec_prefix_proj`
tensors). Those are the only parameters the stage-2 seed initialises.

Training-seed variance at this stage may therefore be small. That is a real finding, and a
reason to test the pre-training seed as a second factor later.

## Running several seeds (the actual study)

Once the pipeline is verified, `ecr_seed.sh` is the script that produces results. It gives
each job its own working directory, so seeds run **concurrently** — the AI-Lab allows 8
GPUs and 8 active jobs per user, one GPU each here.

```bash
for s in 8 1 2; do sbatch ecr_seed.sh $s; done
squeue --me
```

The default is ECR's own recipe: **5 epochs with `--gradient_accumulation_steps 8`**, both
taken from their README's `train_rec.py` command. Accumulation matters twice over — it makes
the effective batch 128 rather than 16, and it divides the optimiser-step count by 8
(265 steps per epoch, 1,325 in total), which is what makes their `--num_warmup_steps 530`
a sensible value instead of a fifth of the whole run. A sweep that quietly dropped it would
be measuring our recipe's seed sensitivity, not ECR's.

Cost: accumulation does not change how many forward/backward passes happen, so an epoch is
still ~41 min, plus a valid+test round of ~8 min. That is **~3 h 45 min per seed** (measured
on seed 8, job 915277: 8.26 s per optimiser step × 1,325 steps), and three seeds run side
by side finish in that same ~3 h 45 min rather than 11 h in sequence. Collect the final test
block from each `slurm-<ID>.out`.

Note there is no baseline evaluation before training: `train_rec.py:271` guards that path
with `if args.test`, and the training loop is its `else`. The first metrics you see are
after epoch 1, so there is no untrained reference point in the log — use `ecr_eval.sh` on
the stage-1 checkpoint if you want one.

What to look at in the output: the spread of `test recall@10` across seeds. If it is much
smaller than the gaps between published baselines, that is a finding about stage 2, not a
null result — see the last section of this file for why the stage-2 seed touches only five
tensors.
