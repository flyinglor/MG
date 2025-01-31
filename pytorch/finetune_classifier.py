import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import unet3d
import nnUNet3d
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
import random
import wandb
from attentive_pooler import AttentiveClassifier


random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


writer = SummaryWriter('logs')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Total CUDA devices: ", torch.cuda.device_count())

config = models_genesis_config()
config.display()

disable_wandb = config.disable_wandb

if not disable_wandb:
    wandb_id = wandb.util.generate_id()
    run = wandb.init(project=f"MG_AttentiveProbing_{config.dataset}", 
                    name=f"{config.pretrain_dsname}_bs{config.batch_size}_ep{config.nb_epoch}_lr{config.lr}_64x64x64", 
                    id=wandb_id,
                    resume='allow',
                    dir=config.logs_path)
    wandb.define_metric("custom_step")

img_transforms = [
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.CropOrPad((128, 128, 128)),
    tio.Resize((64, 64, 64)),#64
]
transform = tio.Compose(img_transforms)

# Define the logger
LOG = logging.getLogger(__name__)

# Configure the logger as needed
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO

fivefolds_train_loss = []
fivefolds_val_loss = []
fivefolds_test_loss = []
fivefolds_test_accuracy = []
fivefolds_test_bal_accuracy = []
fivefold_test_precision = []
fivefold_test_recall = []
fivefold_test_f1 = []

fivefolds_train_loss_last = []
fivefolds_val_loss_last = []
fivefolds_test_loss_last = []
fivefolds_test_accuracy_last = []
fivefolds_test_bal_accuracy_last = []
fivefold_test_precision_last = []
fivefold_test_recall_last = []
fivefold_test_f1_last = []

dataset_name = config.dataset
if dataset_name == 'DZNE':
    data_dir = "/dss/dsshome1/0C/ge79qex2/ModelsGenesis/dataset/DZNE/"
    file_name = "DZNE_CN_FTD_AD.h5"
elif dataset_name == "HOSPITAL":
    data_dir = "/dss/dsshome1/0C/ge79qex2/ModelsGenesis/dataset/238+19+72_tum_splits/"
    file_name = "238+19+72_tum.h5"

# for learning_rate in [1e-2, 1e-3, 1e-4]:
for i in range(1,6):
    print(f"######################### Fold {i} ###########################")

    if not disable_wandb:
        # define which metrics will be plotted against it
        # wandb.define_metric(f"Fold {i} - lr", step_metric="custom_step")
        wandb.define_metric(f"Fold {i} - Training Loss", step_metric="custom_step")
        wandb.define_metric(f"Fold {i} - Validation Loss", step_metric="custom_step")

    if dataset_name == "HOSPITAL":
        # Load data from the .npy files
        train_data = np.load(f'{data_dir}{i}-train.npy', allow_pickle=True)
        test_data = np.load(f'{data_dir}{i}-test.npy', allow_pickle=True)
        val_data = np.load(f'{data_dir}{i}-valid.npy', allow_pickle=True)
    elif dataset_name == 'DZNE':
        # Load data from the .csv files for DZNE
        train_df = pd.read_csv(f'{data_dir}{i}-train.csv')
        #test_df = pd.read_csv(f'{data_dir}test.csv')
        test_df = pd.read_csv(f'{data_dir}{i}-test.csv')
        val_df = pd.read_csv(f'{data_dir}{i}-valid.csv')
        train_data = list(train_df["IMAGEID"])
        test_data = list(test_df["IMAGEID"])
        val_data = list(val_df["IMAGEID"])


    diagnosis = []
    image_train = []
    label_train = []
    image_test = []
    label_test = []
    image_val = []
    label_val = []

    with h5py.File(data_dir+file_name, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            if dataset_name == "HOSPITAL":
                rid = group.attrs['RID']
            elif dataset_name == "DZNE":
                rid = group.attrs["IMAGEID"]
                
            mri_data = group['MRI/T1/data'][:]
            # print(mri_data[np.newaxis, :, :, :].shape)

            # Using torchio
            image_tensor = torch.tensor(mri_data[np.newaxis])
            subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
            transformed_subject = transform(subject)
            transformed_image = transformed_subject['image'].data

            diagnosis.append(group.attrs['DX'])

            if rid in train_data:
                image_train.append(transformed_image)
                label_train.append(group.attrs['DX'])
            elif rid in test_data:
                image_test.append(transformed_image)
                label_test.append(group.attrs['DX'])
            else:
                image_val.append(transformed_image)
                label_val.append(group.attrs['DX'])

    LOG.info("DATASET: %s", data_dir)
    LOG.info("TRAIN: %d", len(label_train))
    LOG.info("TEST: %d", len(label_test))
    LOG.info("VALIDATION: %d", len(label_val))

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    print(OH_encoder.categories_)

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

    # prepare the 3D model
    class TargetNet(nn.Module):
        def __init__(self, base_model,n_class=1):
            super(TargetNet, self).__init__()

            self.base_model = base_model
            # self.dense_1 = nn.Linear(512, 256, bias=True)
            # self.dense_2 = nn.Linear(256, n_class, bias=True)
            # self.dense_1 = nn.Linear(512, n_class, bias=True)
            # self.dense_1 = nn.Linear(256, n_class, bias=True)

            self.classifier = AttentiveClassifier(
                embed_dim=512,
                depth=1,
                num_heads=8,
                num_classes=n_class
            )

        def forward(self, x):
            # --------- fc ------------------------
            # self.base_model(x)
            # # self.base_out = self.base_model.out512
            # self.base_out = self.base_model.out256
            # # This global average polling is for shape (N,C,H,W) not for (N, H, W, C)
            # # where N = batch_size, C = channels, H = height, and W = Width
            # self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
            # # self.linear_out = self.dense_1(self.out_glb_avg_pool)
            # # #add bn
            # # final_out = self.dense_2( F.relu(self.linear_out))
            # final_out = self.dense_1( F.relu(self.out_glb_avg_pool))
            # --------- fc ------------------------
            # --------- attentive pooler ------------------------
            self.base_model(x)
            self.base_out = self.base_model.out512
            # print("self.base_out.shape: ", self.base_out.shape) #size[8, 512, 10, 10, 10]
            self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
            # print("self.out_glb_avg_pool.shape: ", self.out_glb_avg_pool.shape) #torch.Size([8, 512])
            final_out = self.classifier(self.out_glb_avg_pool)

            return final_out
            
    base_model = unet3d.UNet3D()
    # base_model = nnUNet3d.nnUNet3D()

    #Load pre-trained weights
    weight_dir = config.checkpoint_dir
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['state_dict']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    base_model.load_state_dict(unParalled_state_dict)

    # # Freeze the parameters in the base model
    # for param in base_model.parameters():
    #     param.requires_grad = False

    target_model = TargetNet(base_model, n_class=3)
    best_model = TargetNet(base_model, n_class=3)

    # target_model = nn.DataParallel(target_model, device_ids = [i for i in range(torch.cuda.device_count())])
    target_model.to(device)

    print(target_model)

    #three classes in the hospital dataset
    criterion = nn.CrossEntropyLoss()

    # Define the parameters
    learning_rate = config.lr
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Initialize the optimizer with Adam method
    # optimizer = torch.optim.Adam(target_model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
    # optimizer = torch.optim.Adam(target_model.parameters(), 1e3)
    optimizer = torch.optim.Adam(target_model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(target_model.dense_1.parameters(), lr=learning_rate)
    # optimizer.add_param_group({'params': target_model.dense_2.parameters()})

    scheduler_name = "ReduceLROnPlateau"

    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.patience * 0.8), gamma=0.5)
    elif scheduler_name == "LambdaLR":
        # Lambda LR
        lambda1 = lambda epoch: 0.65 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif scheduler_name == "CosineAnnealingLR":
        # CosineAnnealingLR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


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
    iteration = 0
    for epoch in range(intial_epoch, config.nb_epoch):
        scheduler.step(epoch)
        target_model.train()
        for batch_ndx, (x,y) in enumerate(train_loader):
            x, y = x.float().to(device), y.to(device)
            # x, y =  torch.from_numpy(x).float().to(device), y.float().to(device)   
            # pred = F.sigmoid(target_model(x))
            pred = F.softmax(target_model(x), dim=1)
            # print(pred)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(round(loss.item(), 2))

            if not disable_wandb:
                wandb.log(
                    {
                    f"Fold {i} - Training Loss": np.average(train_losses),
                    "custom_step": iteration,
                    },
                )
            if (iteration + 1) % 5 ==0:
                print('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
                    .format(epoch + 1, config.nb_epoch, iteration + 1, np.average(train_losses)))
                # writer.add_scalar('training loss', np.average(train_losses), epoch + 1)
                sys.stdout.flush()

            iteration += 1

        with torch.no_grad():
            target_model.eval()
            print("validating....")
            for batch_ndx, (x,y) in enumerate(val_loader):
                x, y = x.float().to(device), y.to(device)
                # pred = F.sigmoid(target_model(x))
                pred = F.softmax(target_model(x), dim=1)
                loss = criterion(pred, y)
                valid_losses.append(loss.item())

        if not disable_wandb:
            wandb.log(
                {
                f"Fold {i} - Validation Loss": np.average(valid_losses),
                "custom_step": iteration,
                },
            )

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
            best_model = target_model
            best_loss = valid_loss
            best_train_loss = train_loss
            num_epoch_no_improvement = 0
            torch.save({
                'epoch': epoch+1,
                'state_dict' : target_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },os.path.join(config.model_path, "Genesis_ADNI_"+dataset_name+".pt"))
            print("Saving model ",os.path.join(config.model_path, "Genesis_ADNI_"+dataset_name+".pt"))

            # if(len(fivefolds_val_loss)==0):
                #save model
            #     torch.save({
            #         'epoch': epoch+1,
            #         'state_dict' : target_model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict()
            #     },os.path.join(config.model_path, "Genesis_ADNI_Hospital_"+scheduler_name+".pt"))
            #     print("Saving model ",os.path.join(config.model_path, "Genesis_ADNI_Hospital_"+scheduler_name+".pt"))
            # elif (valid_loss < np.min(fivefolds_val_loss)):
            #     #save model
            #     torch.save({
            #         'epoch': epoch+1,
            #         'state_dict' : target_model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict()
            #     },os.path.join(config.model_path, "Genesis_ADNI_Hospital_"+scheduler_name+".pt"))
            #     print("Saving model ",os.path.join(config.model_path, "Genesis_ADNI_Hospital_"+scheduler_name+".pt"))
        else:
            print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,num_epoch_no_improvement))
            num_epoch_no_improvement += 1
        if num_epoch_no_improvement == config.patience:
            print("Early Stopping")
            #save the last model
            # torch.save({
            #     'epoch': epoch+1,
            #     'state_dict' : target_model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict()
            # },os.path.join("/home/hui/ModelsGenesis/pytorch/last_models", "Genesis_ADNI_"+dataset_name+"_"+str(i)+".pt"))
            # print("Saving last model ",os.path.join("/home/hui/ModelsGenesis/pytorch/last_models", "Genesis_ADNI_"+dataset_name+"_"+str(i)+".pt"))
            break
        sys.stdout.flush()

    fivefolds_train_loss.append(best_train_loss)
    fivefolds_val_loss.append(best_loss)
    fivefolds_train_loss_last.append(train_loss)
    fivefolds_val_loss_last.append(valid_loss)

    # Tesing the best model
    test_losses = []
    predictions = []

    with torch.no_grad():
        best_model.eval()
        print("Testing the best model ....")
        for batch_ndx, (x,y) in enumerate(test_loader):
            x, y = x.float().to(device), y.to(device)
            pred = F.softmax(best_model(x), dim=1)
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

    # Tesing the last model
    test_losses = []
    predictions = []

    with torch.no_grad():
        target_model.eval()
        print("Testing the last model ....")
        for batch_ndx, (x,y) in enumerate(test_loader):
            x, y = x.float().to(device), y.to(device)
            pred = F.softmax(target_model(x), dim=1)
            predictions.extend(torch.argmax(pred, dim=1).tolist())
            loss = criterion(pred, y)
            test_losses.append(loss.item())

    print("Test loss:", np.average(test_losses))

    # label_test = torch.argmax(torch.from_numpy(label_test), dim=1).tolist()

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


    fivefolds_test_loss_last.append(np.average(test_losses))
    fivefolds_test_accuracy_last.append(accuracy)
    fivefolds_test_bal_accuracy_last.append(bal_acc)
    fivefold_test_precision_last.append(precision)
    fivefold_test_recall_last.append(recall)
    fivefold_test_f1_last.append(f1)

    print(label_test)
    print(predictions)



# Writing to CSV
with open(os.path.join("metrics/", 'best_adni_1_'+dataset_name+'.csv'), mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Fold', 'Train Loss', 'Val Loss', 'Test Loss', 'Test Accuracy', 'Test Balanced Accuracy', 'Test Precision', 'Test Recall', 'Test F1'])
    for i in range(len(fivefolds_train_loss)):
        csv_writer.writerow([i+1, fivefolds_train_loss[i], fivefolds_val_loss[i], fivefolds_test_loss[i],
                        fivefolds_test_accuracy[i], fivefolds_test_bal_accuracy[i], fivefold_test_precision[i], 
                        fivefold_test_recall[i], fivefold_test_f1[i]])

# Writing to CSV
with open(os.path.join("metrics/", 'last_adni_1_'+dataset_name+'.csv'), mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Fold', 'Train Loss', 'Val Loss', 'Test Loss', 'Test Accuracy', 'Test Balanced Accuracy', 'Test Precision', 'Test Recall', 'Test F1'])
    for i in range(len(fivefolds_train_loss)):
        csv_writer.writerow([i+1, fivefolds_train_loss_last[i], fivefolds_val_loss_last[i], fivefolds_test_loss_last[i],
                        fivefolds_test_accuracy_last[i], fivefolds_test_bal_accuracy_last[i], fivefold_test_precision_last[i], 
                        fivefold_test_recall_last[i], fivefold_test_f1_last[i]])

print("Averages for the best model")
# print("Learning rate: ", learning_rate)
print(f"Average Train Loss: {np.mean(fivefolds_train_loss)}, std: {np.std(fivefolds_train_loss)}")
print(f"Average Val Loss: {np.mean(fivefolds_val_loss)}, std: {np.std(fivefolds_val_loss)}")
print(f"Average Test Loss: {np.mean(fivefolds_test_loss)}, std: {np.std(fivefolds_test_loss)}")
print(f"Average Test Accuracy: {np.mean(fivefolds_test_accuracy)}, std: {np.std(fivefolds_test_accuracy)}")
print(f"Average Test Balanced Accuracy: {np.mean(fivefolds_test_bal_accuracy)}, std: {np.std(fivefolds_test_bal_accuracy)}")
print(f"Average Test Precision: {np.mean(fivefold_test_precision)}, std: {np.std(fivefold_test_precision)}")
print(f"Average Test Recall: {np.mean(fivefold_test_recall)}, std: {np.std(fivefold_test_recall)}")
print(f"Average Test F1: {np.mean(fivefold_test_f1)}, std: {np.std(fivefold_test_f1)}")

print("Averages for the last model")
# print("Learning rate: ", learning_rate)
print(f"Average Train Loss: {np.mean(fivefolds_train_loss_last)}, std: {np.std(fivefolds_train_loss_last)}")
print(f"Average Val Loss: {np.mean(fivefolds_val_loss_last)}, std: {np.std(fivefolds_val_loss_last)}")
print(f"Average Test Loss: {np.mean(fivefolds_test_loss_last)}, std: {np.std(fivefolds_test_loss_last)}")
print(f"Average Test Accuracy: {np.mean(fivefolds_test_accuracy_last)}, std: {np.std(fivefolds_test_accuracy_last)}")
print(f"Average Test Balanced Accuracy: {np.mean(fivefolds_test_bal_accuracy_last)}, std: {np.std(fivefolds_test_bal_accuracy_last)}")
print(f"Average Test Precision: {np.mean(fivefold_test_precision_last)}, std: {np.std(fivefold_test_precision_last)}")
print(f"Average Test Recall: {np.mean(fivefold_test_recall_last)}, std: {np.std(fivefold_test_recall_last)}")
print(f"Average Test F1: {np.mean(fivefold_test_f1_last)}, std: {np.std(fivefold_test_f1_last)}")

writer.close()