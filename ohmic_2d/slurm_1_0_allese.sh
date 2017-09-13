#!/bin/sh  
#SBATCH --job-name=heom_2d_L-1_K-0_allese
#SBATCH --output=heom_2d_L-1_K-0_allese.log
#SBATCH --error=heom_2d_L-1_K-0_allese.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH -t 24:00:00

WORKDIR=$SLURM_SUBMIT_DIR
cd $WORKDIR

python -u driver.py 1 0 allese
