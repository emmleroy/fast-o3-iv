#!/bin/bash
#SBATCH --job-name=apply_emissions_filter-cesm2wa6-ssp245-CO_anthro-downwind
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eleroy@mit.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1       
#SBATCH --mem-per-cpu=20G       
#SBATCH --time=6:00:00 # large r for CO takes longer to run
#SBATCH -p hdr
#SBATCH --output=slurm_output/%j_%x.out
#SBATCH --error=slurm_output/%j_%x.err

labels=(
  001 002 003 004
  005 006 007 008
  009 010 011 012
  013 014 015 016
)

for label in "${labels[@]}"; do
  srun python apply_emissions_filter.py \
    -l "$label" -m CESM2-WACCM6 -s ssp245 -v CO_em_anthro -o downwind
done
