#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import torch
from torchsummary import summary
import sys
from utils import *
import unet3d
from config import models_genesis_config
from tqdm import tqdm
import torchio as tio
import h5py
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

def plot_and_save_images(imgs, gt, save_path="output.png"):
    """
    Plot and save middle slices of imgs, contexts, and targets for visualization.
    """
    # Ensure dimensions are correct and find the middle slice index
    middle_index = 16  # Middle slice along depth

    batch_size = len(contexts)
    fig, axes = plt.subplots(batch_size, 2, figsize=(15, batch_size * 2))
    axes = np.atleast_2d(axes)

    for i in range(batch_size):
        ax = axes[i]

        # Plot middle slices
        ax[0].imshow(imgs[i, :, :, middle_index], cmap="gray")
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(gt[i, :, middle_index], cmap="gray")
        ax[1].set_title("Context")
        ax[1].axis("off")

    # Adjust row spacing
    plt.subplots_adjust(hspace=0.1)  # Decrease hspace for less space between rows

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

conf = models_genesis_config()
conf.display()

# train_dir = "/dss/dsshome1/0C/ge79qex2/ModelsGenesis/generated_cubes/ADNI/train_3_64x64x32.npy"
val_dir = "/dss/dsshome1/0C/ge79qex2/ModelsGenesis/generated_cubes/ADNI/valid_64x64x32.npy"

# x_train = np.load(train_dir)
# x_train = np.expand_dims(np.array(x_train), axis=1)
# print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))
x_valid = np.load(val_dir)
x_valid = np.expand_dims(np.array(x_valid), axis=1)
print("x_valid: {} | {:.2f} ~ {:.2f}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))

batch_size=4
# training_generator = generate_pair(x_train, batch_size, conf)
validation_generator = generate_pair(x_valid, batch_size, conf)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image, gt = next(validation_generator)
gt = np.repeat(gt,conf.nb_class,axis=1)
image, gt = torch.from_numpy(image).float().to(device), torch.from_numpy(gt).float().to(device)
plot_and_save_images(image, gt)