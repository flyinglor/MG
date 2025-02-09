#!/bin/bash
#SBATCH -J adni
#SBATCH -N 1
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --mem=40gb
#SBATCH --ntasks=1
#SBATCH --mail-user=hui.zheng@tum.de
#SBATCH --mail-type=ALL
#SBATCH --time=72:00:00
#SBATCH -o %x.%j.%N.out
#SBATCH -e %x.%j.%N.err

source ~/.bashrc  # activate miniconda
source ~/miniconda3/bin/activate MG # activate your environment

cd ~/ModelsGenesis/pytorch/ 

srun python Genesis_ADNI.py