#!/bin/bash
#SBATCH --job-name=student-test
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=500M
#SBATCH --cpus-per-task=8
#SBATCH --output=../%x-%j.out
#SBATCH --error=../%x-%j.err
#SBATCH --mail-type=END,FAIL


module load Anaconda3/2022.10

eval "$(conda shell.bash hook)"

conda activate mimic

srun python multitype.py
