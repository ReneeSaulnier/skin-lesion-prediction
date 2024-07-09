# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
import matplotlib.pyplot as plt
import keras
from keras import layers, models, optimizers
import h5py
from PIL import Image
import io

# # Pipeline

# ### Data Collection

text_data_df = pd.read_csv('data/isic-2024-challenge/train-metadata.csv')
text_data_df.head()

# ### EDA for the text data

# +
# Disribution of the target variable

# 0 - Benign
# 1 - Malignant
text_data_df['target'].value_counts()

malignant_df = text_data_df[text_data_df['target'] == 1]

# +
# Group the data by sex and target (benign or malignant)
grouped_data = text_data_df.groupby(['sex', 'target']).size().reset_index(name='count')

# Plot the grouped data
fig = px.bar(grouped_data, x='sex', y='count', color='target', barmode='group', 
             labels={'sex': 'Sex', 'count': 'Count', 'target': 'Lesion Type'},
             title='Distribution of the sexes')

fig.show()

# +
# See if there is a correlation between the target variable and the anatom_site_general
grouped_data = malignant_df.groupby(['anatom_site_general', 'target']).size().reset_index(name='count')

fig = px.pie(grouped_data, values='count', names='anatom_site_general', title='Distribution of the anatomical sites where a malignant lesion is found')

fig.show()
# -

# Analyze the age correlation with the target variable
fig = px.histogram(malignant_df, x='age_approx', title='Distribution of the age of the patients with malignant lesions')
fig.show()


train_df = pd.read_csv('data/isic-2024-challenge/train-metadata.csv')
test_df = pd.read_csv('data/isic-2024-challenge/test-metadata.csv')


# ### Data Processing

# +
# Load the training datasets
image_train_ds = tf.keras.utils.image_dataset_from_directory(
    'data/isic-2024-challenge/train-image',
    seed=123,
    batch_size=32,
    validation_split=0.2,
    subset="training",
    shuffle=True
)

image_val_ds = tf.keras.utils.image_dataset_from_directory(
    'data/isic-2024-challenge/train-image',
    seed=123,
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    shuffle=True
)

# Load the test datasets
file_path = 'data/isic-2024-challenge/test-image/image/test-image.hdf5'
image_test_ds = []

with h5py.File(file_path, "r") as f:
    dataset_names = ['ISIC_0015657', 'ISIC_0015729', 'ISIC_0015740']
    
    for dataset_name in dataset_names:
        image_data_bytes = f[dataset_name][()]
        image = Image.open(io.BytesIO(image_data_bytes))
        image_array = np.array(image)
        image_test_ds.append(image_array)
# -

# Scale the image data
image_train_ds = image_train_ds.map(lambda x, y: (x / 255, y))
data_it = image_train_ds.as_numpy_iterator()
batch = data_it.next()

# ### Model Training

# +

model = models.Sequential([

    # 3 convolutional layers with max pooling
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    
    # Flatten the output of the last convolutional/pooling layer
    layers.Flatten(),
    
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(image_train_ds, epochs=20, validation_data=image_val_ds)
# -

# ### Model Validation
