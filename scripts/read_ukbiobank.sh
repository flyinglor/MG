#!/bin/bash
#SBATCH -J read_ukbiobank
#SBATCH --mem=512gb
#SBATCH --ntasks=1
#SBATCH --mail-user=hui.zheng@tum.de
#SBATCH --mail-type=ALL
#SBATCH --time=72:00:00
#SBATCH -o %x.%j.%N.out
#SBATCH -e %x.%j.%N.err

source ~/.bashrc  # activate miniconda
source ~/miniconda3/bin/activate MG # activate your environment

cd ~/ModelsGenesis/

srun python ukbiobank.py