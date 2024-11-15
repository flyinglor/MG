import os
import h5py
from collections import Counter
import numpy as np

# Directory path
root_path = '/dss/dssmcmlfs01/pr62la/pr62la-dss-0002/MSc/Hui/UKB_CAT12'

# List all files in directory

suffix = 'uk_valid.h5' 
data_dir = os.path.join(root_path, suffix)
diagnosis = []
image_train = []
with h5py.File(data_dir, mode='r') as file:
    for name, group in file.items():
        # print(f"Group: {name}")
        # # Access and print all the attributes of the group
        # for attr_name, attr_value in group.attrs.items():
        #     print(f"Attribute Name: {attr_name}, Attribute Value: {attr_value}")

        if name == "stats":
            continue
        # rid = group.attrs['RID']
        mri_data = group['MRI/T1/data'][:]
        print(mri_data[np.newaxis, :, :, :].shape)
        transformed_image = mri_data[np.newaxis, :, :, :]

        # diagnosis.append(group.attrs['DX'])
        image_train.append(transformed_image)

print("Validation set: ")
print(len(image_train))
print(image_train[0].shape)

# counter = Counter(diagnosis)
# # Print unique values and their counts
# for value, count in counter.items():
#     print(f"Value: {value}, Count: {count}")

# suffix = 'uk_train.h5' 
# data_dir = os.path.join(root_path, suffix)
# diagnosis = []
# image_train = []
# with h5py.File(data_dir, mode='r') as file:
#     for name, group in file.items():
#         if name == "stats":
#             continue
#         # rid = group.attrs['RID']
#         mri_data = group['MRI/T1/data'][:]
#         # print(mri_data[np.newaxis, :, :, :].shape)
#         transformed_image = mri_data[np.newaxis, :, :, :]

#         # diagnosis.append(group.attrs['DX'])
#         image_train.append(transformed_image)

# print("Train set: ")
# print(len(image_train))
# print(image_train[0].shape)

# counter = Counter(diagnosis)
# # Print unique values and their counts
# for value, count in counter.items():
#     print(f"Value: {value}, Count: {count}")

# def get_ukb_dataset(root_path, train=True, train_transform=None):
#     suffix = 'train.h5' if train else 'valid.h5'
#     data_dir = os.path.join(root_path, suffix)
#     diagnosis = []
#     image_train = []
#     label_train = []
#     with h5py.File(data_dir, mode='r') as file:
#         for name, group in file.items():
#             if name == "stats":
#                 continue
#             rid = group.attrs['RID']
#             mri_data = group['MRI/T1/data'][:]
#             # print(mri_data[np.newaxis, :, :, :].shape)

#             if train_transform:
#                 # Using torchio
#                 image_tensor = torch.tensor(mri_data[np.newaxis])
#                 subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
#                 transformed_subject = train_transform(subject)
#                 transformed_image = transformed_subject['image'].data
#             else:
#                 transformed_image = mri_data[np.newaxis, :, :, :]

#             diagnosis.append(group.attrs['DX'])
#             image_train.append(transformed_image)
#             label_train.append(group.attrs['DX'])

#     OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#     OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
#     label_train = OH_encoder.transform(np.array(label_train).reshape(-1, 1))
#     train_set = CustomImageDataset(image_train, label_train)
#     return train_set