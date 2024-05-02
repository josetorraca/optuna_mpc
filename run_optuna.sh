#!/bin/bash
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -l walltime=24:00:00
#PBS -N run_optuna

cd $PBS_O_WORKDIR

module load tools/prod
# module load SciPy-bundle/2022.05-foss-2022a

python3 optuna_nmpc_base_v2.py