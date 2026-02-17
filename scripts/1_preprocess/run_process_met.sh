#!/bin/bash
#SBATCH --job-name=run_process_met-cesm2wa6-ssp245-V
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eleroy@mit.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1       
#SBATCH --mem-per-cpu=20G       
#SBATCH --time=6:00:00 
#SBATCH -p hdr
#SBATCH --output=slurm_output/%j_%x.out
#SBATCH --error=slurm_output/%j_%x.err

labels=(
  001 002 003 004
  005 006 007 008
  009 010 #011 012
  #013 014 015 016
)

for label in "${labels[@]}"; do
  python run_process_met.py -l "$label" -m CESM2-WACCM6 -s ssp245 -v V -o surface
done
