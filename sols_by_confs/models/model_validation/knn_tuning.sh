#!/usr/bin/env bash
#SBATCH --job-name=tuning_for_knn_models
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=50:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute

echo "Job start at $(date)"
python /work/luduslab/sts_3sat/sols_by_confs/models/KNNs.py
echo "Job end at $(date)"
