#!/bin/bash
#SBATCH -J MG_adni_atp_dzne_bs8_ep100_seed0_64x64x64_lr5e5
#SBATCH -N 1
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --ntasks=1
#SBATCH --mail-user=hui.zheng@tum.de
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH -o %x.%j.%N.out

source ~/.bashrc  # activate miniconda
source ~/miniconda3/bin/activate MG # activate your environment

cd ~/ModelsGenesis/pytorch/ 

srun python finetune_classifier.py