#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=1-1:00:00  # max job runtime
#SBATCH --cpus-per-task=1  # number of processor cores
#SBATCH --nodes=1  # number of nodes
#SBATCH --partition=gpu  # partition(s)
#SBATCH --gres=gpu:1
#SBATCH --mem=1G  # max memory
#SBATCH -J "Embedding-Esm"  # job name
#SBATCH --mail-user=maramos@iastate.edu  # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


python test.py