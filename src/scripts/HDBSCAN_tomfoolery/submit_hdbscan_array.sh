

#!/bin/bash
#SBATCH --job-name=hdbscan_gs_fullmatrix
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --array=0-399
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

module load <PYTHON_MODULE_OR_CONDA_MODULE?
conda activate HDBSCAN_GS

python src/scripts/HDBSCAN_fullmatrix_slurm.py
