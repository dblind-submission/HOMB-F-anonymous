#!/bin/bash
#SBATCH --job-name=student-test
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=500M
#SBATCH --cpus-per-task=8
#SBATCH --output=../%x-%j.out
#SBATCH --error=../%x-%j.err
#SBATCH --mail-type=END,FAIL

srun python multitype.py
