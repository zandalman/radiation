#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH -t 3:00 # 3 minutes
#SBATCH --mem=1g #1 GB of memeory

module load miniconda
conda activate /gpfs/loomis/project/phys678/conda_envs/phys678
python RTE.py
