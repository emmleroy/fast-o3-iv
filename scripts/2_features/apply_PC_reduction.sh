#!/bin/bash
#SBATCH --job-name=apply_PC_reduction-cesm2-ssp245-zg500
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eleroy@mit.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1       
#SBATCH --mem-per-cpu=20G       
#SBATCH --time=6:00:00 # large r for CO takes longer to run
#SBATCH -p hdr
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --error=slurm_output/%x_%j.err

srun python apply_PC_reduction.py -m CESM2 -s ssp245 -v zg500 -t 001
