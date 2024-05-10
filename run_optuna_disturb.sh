#!/bin/bash
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -l walltime=24:00:00
#PBS -N run_optuna
#PBS -o job_output.txt
#PBS -e job_error.txt

# Display current directory and its contents
echo "Current directory: $(pwd)"
echo "Listing contents of the current directory:"
ls -l

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

# Load the production tools environment
module load tools/prod

# Initialize Conda from the base environment
source /apps/jupyterhub/2019-04-29/miniconda/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate pcgym310

python3 optuna_nmpc_disturb.py

# Optional: Check if the script is producing output correctly
echo "Python script completed"
