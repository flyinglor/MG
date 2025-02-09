#!/bin/bash
#SBATCH -J generate_cubes
#SBATCH -N 1
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --mem=64gb
#SBATCH --ntasks=1
#SBATCH --mail-user=hui.zheng@tum.de
#SBATCH --mail-type=ALL
#SBATCH --time=72:00:00
#SBATCH -o %x.%j.%N.out
#SBATCH -e %x.%j.%N.err

source ~/.bashrc  # activate miniconda
source ~/miniconda3/bin/activate MG # activate your environment

cd ~/ModelsGenesis/

srun python -W ignore infinite_generator_3D.py --scale 3 --data dataset/ADNI --save generated_cubes/ADNI