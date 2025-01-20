#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -c 1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=MLGliomaSegmentation
#SBATCH --output=MLGliomaSegmentation.out
#SBATCH --error=MLGliomaSegmentation.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniele.ferrario.001@student.uni.lu


cd /scratch/users/dferrario/MLGliomaSegmentation
micromamba activate glioma
python src/main.py