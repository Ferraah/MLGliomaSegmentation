# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random
import nibabel

sns.set()
from dataloader import get_data_path, get_data_loader

# Load the data
data_path = get_data_path()

training_data_path = os.path.join(data_path, "archive", "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData")
csv_train_path = os.path.join(training_data_path, "survival_info.csv")
train_df = pd.read_csv(csv_train_path)

# print(train_df.head())
# print(train_df.info())

# print(f"The total patient ids are {train_df['Brats20ID'].count()}, from those the unique ids are {train_df['Brats20ID'].value_counts().shape[0]} ")

patients_count = train_df['Brats20ID'].value_counts()

random_patient_id = "010"

patient_path = os.path.join(training_data_path, f"BraTS20_Training_{random_patient_id}")

image_path = os.path.join(patient_path, f"BraTS20_Training_{random_patient_id}_t1.nii") # Replace with actual path
label_path = os.path.join(patient_path, f"BraTS20_Training_{random_patient_id}_flair.nii") # Replace with actual path

nifti_dataset = nibabel.load(label_path)
nifti_images = nifti_dataset.get_fdata()

image = nifti_images[:, :, 80]
plt.imshow(image)
plt.show()