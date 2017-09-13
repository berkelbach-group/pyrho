#!/bin/sh  
#SBATCH --job-name=heom_2d_L-4_K-0_allgsb
#SBATCH --output=heom_2d_L-4_K-0_allgsb.log
#SBATCH --error=heom_2d_L-4_K-0_allgsb.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --partition=sandyb
#SBATCH --qos=sandyb-long
#SBATCH --time=7-00:00:00

WORKDIR=$SLURM_SUBMIT_DIR
cd $WORKDIR

python -u driver.py 4 0 allgsb
