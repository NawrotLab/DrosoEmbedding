#!/bin/bash
#SBATCH --job-name=ko_permutation
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=agmn-srv-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --mail-user=abdelbaki.amina@icloud.com

source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate

cd /rhomes/aabdel/DrosoEmbedding

# python -m scripts.diagnostics.diag_static_fill_viz                      # baseline fill diagnostic
# python -m scripts.diagnostics.diag_KO_quality --variant shuffled
# python -m scripts.diagnostics.diag_KO_permutation --variant shuffled
# python -m scripts.diagnostics.diag_KO_quality --variant static
# python -m scripts.diagnostics.diag_KO_permutation --variant static
# python -m scripts.diagnostics.diag_KO_quality --variant noisefill
# python -m scripts.diagnostics.diag_KO_permutation --variant noisefill
# python -m scripts.diagnostics.diag_KO_quality                           # zero-fill
# python -m scripts.diagnostics.diag_KO_permutation                       # zero-fill
