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
import plotly.express as px
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, models

# # Pipeline

# ### Data Collection

# Explore the text data
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


# ### EDA for the image data

# +
# Load the training dataset
image_train_ds = tf.keras.utils.image_dataset_from_directory(
    'data/isic-2024-challenge/train-image',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(256, 256),
    batch_size=32
)

# Load the validation dataset
image_val_ds = tf.keras.utils.image_dataset_from_directory(
    'data/isic-2024-challenge/train-image',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(256, 256),
    batch_size=32
)
# -

# ### Data Processing

# - ##### Preprocess Text Data

# - #### Preprocess Image Data

# +

data_it = image_train_ds.as_numpy_iterator()
image_data_batch = data_it.next()
# -

# Scale the image data
image_data_batch = image_data_batch.map(lambda x, y: (x / 255, y))

# ### Model Training

# ### Model Validation
