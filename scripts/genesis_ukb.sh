#!/bin/bash
#SBATCH -J MG_ukb
#SBATCH -N 1
#SBATCH -p mcml-hgx-h100-92x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --ntasks=1
#SBATCH --mail-user=hui.zheng@tum.de
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00
#SBATCH -o %x.%j.%N.out
#SBATCH -e %x.%j.%N.err

source ~/.bashrc  # activate miniconda
source ~/miniconda3/bin/activate MG # activate your environment

cd ~/ModelsGenesis/pytorch/ 

export WANDB_API_KEY=9b379393a7a65969e05ab4e01683be3b8770aabf


srun python Genesis_ukb.py