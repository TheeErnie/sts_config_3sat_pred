#!/usr/bin/env bash
#SBATCH --job-name=compute_on_all_training_cnf_files
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=35:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=11
#SBATCH --partition=compute

echo "Job start at $(date)"
python /work/luduslab/sts_3sat/sols_by_confs/cnf_data/gen_training_data.py 0
python /work/luduslab/sts_3sat/sols_by_confs/cnf_data/gen_training_data.py 1
python /work/luduslab/sts_3sat/sols_by_confs/cnf_data/gen_training_data.py 2
python /work/luduslab/sts_3sat/sols_by_confs/cnf_data/gen_training_data.py 3
python /work/luduslab/sts_3sat/sols_by_confs/cnf_data/gen_training_data.py 4
echo "Job end at $(date)"
