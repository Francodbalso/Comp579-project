#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=winter2025-comp579 

module load miniconda/miniconda-winter2025

python ../hover_oc.py

python ../plot.py
