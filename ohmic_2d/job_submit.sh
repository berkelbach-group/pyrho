#!/bin/bash

# Check for proper number of command line args.

EXPECTED_ARGS=3

if [ $# -ne $EXPECTED_ARGS ]; then
    echo "Usage: `basename $0` L K lioupath" 
    exit 99
fi

slurmsh="slurm_${1}_${2}_${3}.sh"
sed "s/LLL/$1/g; s/KKK/$2/g; s/LIOU/$3/g;" good_slurm_template.sh > $slurmsh
sbatch $slurmsh 
