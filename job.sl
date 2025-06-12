#!/bin/bash
#SBATCH --partition=sapphire
#SBATCH --job-name=student-test
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=500M
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/gpfs/projects/punim1824/%u/paper2/student-performance/plots/%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim1824/%u/paper2/student-performance/plots/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mojgan.kouhounestani@student.unimelb.edu.au


module load Anaconda3/2022.10

eval "$(conda shell.bash hook)"

conda activate mimic

srun python multitype.py
