#!/bin/bash
#SBATCH -J ukb_generate_cubes
#SBATCH -N 1
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --mem=512gb
#SBATCH --ntasks=1
#SBATCH --mail-user=hui.zheng@tum.de
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00
#SBATCH -o %x.%j.%N.out
#SBATCH -e %x.%j.%N.err

source ~/.bashrc  # activate miniconda
source ~/miniconda3/bin/activate MG # activate your environment

cd ~/ModelsGenesis/

srun python -W ignore infinite_generator_3D.py --scale 1 --data /dss/dssmcmlfs01/pr62la/pr62la-dss-0002/MSc/Hui/UKB_CAT12 --save /dss/dssmcmlfs01/pr62la/pr62la-dss-0002/MSc/Hui/

