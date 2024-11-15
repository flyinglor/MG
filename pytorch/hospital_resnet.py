import resnet
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import unet3d
import os
import sys
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import pandas as pd
import logging
from prepare_data import CustomImageDataset
from config import models_genesis_config
import torchio as tio
from sklearn.preprocessing import OneHotEncoder
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import csv
# import random

# random.seed(0)
# torch.manual_seed(0)
# np.random.seed(0)

writer = SummaryWriter('logs')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Total CUDA devices: ", torch.cuda.device_count())

config = models_genesis_config()
config.display()

# Define the logger
LOG = logging.getLogger(__name__)

# Configure the logger as needed
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO

#Initialize model
model_depth = 18
input_W = 113
input_H = 137
input_D = 113
resnet_shortcut = "B"
no_cuda = False
num_classes = 3

assert model_depth in [10, 18, 34, 50, 101, 152, 200]

if model_depth == 10:
    model = resnet.resnet10(
        sample_input_W=input_W,
        sample_input_H=input_H,
        sample_input_D=input_D,
        shortcut_type=resnet_shortcut,
        no_cuda=no_cuda,
        num_classes=num_classes)
elif model_depth == 18:
    model = resnet.resnet18(
        sample_input_W=input_W,
        sample_input_H=input_H,
        sample_input_D=input_D,
        shortcut_type=resnet_shortcut,
        no_cuda=no_cuda,
        num_classes=num_classes)
elif model_depth == 34:
    model = resnet.resnet34(
        sample_input_W=input_W,
        sample_input_H=input_H,
        sample_input_D=input_D,
        shortcut_type=resnet_shortcut,
        no_cuda=no_cuda,
        num_classes=num_classes)
elif model_depth == 50:
    model = resnet.resnet50(
        sample_input_W=input_W,
        sample_input_H=input_H,
        sample_input_D=input_D,
        shortcut_type=resnet_shortcut,
        no_cuda=no_cuda,
        num_classes=num_classes)
elif model_depth == 101:
    model = resnet.resnet101(
        sample_input_W=input_W,
        sample_input_H=input_H,
        sample_input_D=input_D,
        shortcut_type=resnet_shortcut,
        no_cuda=no_cuda,
        num_classes=num_classes)
elif model_depth == 152:
    model = resnet.resnet152(
        sample_input_W=input_W,
        sample_input_H=input_H,
        sample_input_D=input_D,
        shortcut_type=resnet_shortcut,
        no_cuda=no_cuda,
        num_classes=num_classes)
elif model_depth == 200:
    model = resnet.resnet200(
        sample_input_W=input_W,
        sample_input_H=input_H,
        sample_input_D=input_D,
        shortcut_type=resnet_shortcut,
        no_cuda=no_cuda,
        num_classes=num_classes)

print(model)

fivefolds_train_loss = []
fivefolds_val_loss = []
fivefolds_test_loss = []
fivefolds_test_accuracy = []
fivefolds_test_bal_accuracy = []
fivefold_test_precision = []
fivefold_test_recall = []
fivefold_test_f1 = []

data_dir = "/dss/dsshome1/0C/ge79qex2/ModelsGenesis/dataset/238+19+72_tum_splits/"
for i in range(1,6):
    # Load data from the .npy files
    train_data = np.load(f'{data_dir}{i}-train.npy', allow_pickle=True)
    test_data = np.load(f'{data_dir}{i}-test.npy', allow_pickle=True)
    val_data = np.load(f'{data_dir}{i}-valid.npy', allow_pickle=True)

    diagnosis = []
    image_train = []
    label_train = []
    image_test = []
    label_test = []
    image_val = []
    label_val = []

    with h5py.File(data_dir+"238+19+72_tum.h5", mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            rid = group.attrs['RID']
            mri_data = group['MRI/T1/data'][:]
            diagnosis.append(group.attrs['DX'])
            if rid in train_data:
                image_train.append(mri_data[np.newaxis])
                label_train.append(group.attrs['DX'])
            elif rid in test_data:
                image_test.append(mri_data[np.newaxis])
                label_test.append(group.attrs['DX'])
            else:
                image_val.append(mri_data[np.newaxis])
                label_val.append(group.attrs['DX'])

    LOG.info("DATASET: %s", data_dir)
    LOG.info("TRAIN: %d", len(label_train))
    LOG.info("TEST: %d", len(label_test))
    LOG.info("VALIDATION: %d", len(label_val))

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))

    label_train = OH_encoder.transform(np.array(label_train).reshape(-1, 1))
    label_test = OH_encoder.transform(np.array(label_test).reshape(-1, 1))
    label_val = OH_encoder.transform(np.array(label_val).reshape(-1, 1))

    train_set = CustomImageDataset(image_train, label_train)
    test_set = CustomImageDataset(image_test, label_test)
    val_set = CustomImageDataset(image_val, label_val)

    # prepare your own data
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

    #Training
    model.to(device)

    #three classes in the hospital dataset
    criterion = nn.CrossEntropyLoss()

    # Define the parameters
    learning_rate = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Initialize the optimizer with Adam method
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-3, nesterov=False)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.patience * 0.8), gamma=0.5)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    best_loss = 100000
    intial_epoch =0
    num_epoch_no_improvement = 0

    # train the model
    for epoch in range(intial_epoch, config.nb_epoch):
        scheduler.step(epoch)
        model.train()
        iteration = 0
        for batch_ndx, (x,y) in enumerate(train_loader):
            x, y = x.float().to(device), y.to(device)
            # x, y =  torch.from_numpy(x).float().to(device), y.float().to(device)   
            # pred = F.sigmoid(model(x))
            pred = F.softmax(model(x), dim=1)
            # print(pred)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(round(loss.item(), 2))
            if (iteration + 1) % 5 ==0:
                print('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
                    .format(epoch + 1, config.nb_epoch, iteration + 1, np.average(train_losses)))
                sys.stdout.flush()
            iteration += 1

        with torch.no_grad():
            model.eval()
            print("validating....")
            for batch_ndx, (x,y) in enumerate(val_loader):
                x, y = x.float().to(device), y.to(device)
                # pred = F.sigmoid(model(x))
                pred = F.softmax(model(x), dim=1)
                loss = criterion(pred, y)
                valid_losses.append(loss.item())

        #logging
        train_loss=np.average(train_losses)
        valid_loss=np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1,valid_loss,train_loss))
        writer.add_scalar('training loss', train_loss, epoch + 1)
        writer.add_scalar('validation loss', valid_loss, epoch + 1)

        train_losses=[]
        valid_losses=[]

        if valid_loss < best_loss:
            print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            best_train_loss = train_loss
            num_epoch_no_improvement = 0
            if(len(fivefolds_val_loss)==0):
                #save model
                torch.save({
                    'epoch': epoch+1,
                    'state_dict' : model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                },os.path.join(config.model_path, f"Hospital_Resnet_{str(model_depth)}.pt"))
                print("Saving model ",os.path.join(config.model_path, f"Hospital_Resnet_{str(model_depth)}.pt"))
            elif (valid_loss < np.min(fivefolds_val_loss)):
                #save model
                torch.save({
                    'epoch': epoch+1,
                    'state_dict' : model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                },os.path.join(config.model_path, f"Hospital_Resnet_{str(model_depth)}.pt"))
                print("Saving model ",os.path.join(config.model_path, f"Hospital_Resnet_{str(model_depth)}.pt"))
        else:
            print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,num_epoch_no_improvement))
            num_epoch_no_improvement += 1
        if num_epoch_no_improvement == config.patience:
            print("Early Stopping")
            break
        sys.stdout.flush()

    fivefolds_train_loss.append(best_train_loss)
    fivefolds_val_loss.append(best_loss)


    test_losses = []
    predictions = []

    with torch.no_grad():
        model.eval()
        print("testing....")
        for batch_ndx, (x,y) in enumerate(test_loader):
            x, y = x.float().to(device), y.to(device)
            pred = F.softmax(model(x), dim=1)
            predictions.extend(torch.argmax(pred, dim=1).tolist())
            loss = criterion(pred, y)
            test_losses.append(loss.item())

    print("Test loss:", np.average(test_losses))

    label_test = torch.argmax(torch.from_numpy(label_test), dim=1).tolist()

    accuracy = accuracy_score(label_test, predictions)
    bal_acc = balanced_accuracy_score(label_test, predictions)
    precision = precision_score(label_test, predictions, average='macro')
    recall = recall_score(label_test, predictions, average='macro')
    f1 = f1_score(label_test, predictions, average='macro')

    print("Accuracy:", accuracy)
    print("Balanced Accuracy:", bal_acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    fivefolds_test_loss.append(np.average(test_losses))
    fivefolds_test_accuracy.append(accuracy)
    fivefolds_test_bal_accuracy.append(bal_acc)
    fivefold_test_precision.append(precision)
    fivefold_test_recall.append(recall)
    fivefold_test_f1.append(f1)

    print(label_test)
    print(predictions)

# Writing to CSV
with open(f'metric_hos_resnet_{str(model_depth)}.csv', mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Fold', 'Train Loss', 'Val Loss', 'Test Loss', 'Test Accuracy', 'Test Balanced Accuracy', 'Test Precision', 'Test Recall', 'Test F1'])
    for i in range(len(fivefolds_train_loss)):
        csv_writer.writerow([i+1, fivefolds_train_loss[i], fivefolds_val_loss[i], fivefolds_test_loss[i],
                         fivefolds_test_accuracy[i], fivefolds_test_bal_accuracy[i], fivefold_test_precision[i], 
                         fivefold_test_recall[i], fivefold_test_f1[i]])

print(f"Average Train Loss: {np.mean(fivefolds_train_loss)}, std: {np.std(fivefolds_train_loss)}")
print(f"Average Val Loss: {np.mean(fivefolds_val_loss)}, std: {np.std(fivefolds_val_loss)}")
print(f"Average Test Loss: {np.mean(fivefolds_test_loss)}, std: {np.std(fivefolds_test_loss)}")
print(f"Average Test Accuracy: {np.mean(fivefolds_test_accuracy)}, std: {np.std(fivefolds_test_accuracy)}")
print(f"Average Test Balanced Accuracy: {np.mean(fivefolds_test_bal_accuracy)}, std: {np.std(fivefolds_test_bal_accuracy)}")
print(f"Average Test Precision: {np.mean(fivefold_test_precision)}, std: {np.std(fivefold_test_precision)}")
print(f"Average Test Recall: {np.mean(fivefold_test_recall)}, std: {np.std(fivefold_test_recall)}")
print(f"Average Test F1: {np.mean(fivefold_test_f1)}, std: {np.std(fivefold_test_f1)}")

writer.close()
