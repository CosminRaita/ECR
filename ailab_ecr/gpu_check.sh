#!/bin/bash
#SBATCH --job-name=ecr-gpu-check
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out
# Note: in Juan's email one example says "singularity --nv exec"; the correct order is "exec --nv".
singularity exec --nv ecr.sif python gpu_check.py
