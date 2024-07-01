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

import os
import pandas as pd
import plotly.express as px

# # Pipeline

# ### Data Collection

# Explore the text data
text_data_df = pd.read_csv('data/isic-2024-challenge/train-metadata.csv')
text_data_df.head()

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


# ### EDA

# ### Data Processing

# ### Model Training

# ### Model Validation
