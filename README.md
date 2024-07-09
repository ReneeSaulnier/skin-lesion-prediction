# ISIC 2024 - Skin Cancer Detection with 3D-TBP

## Introduction
This repository contains the code for for the [ISIC 2024 - Skin Cancer Detection with 3D-TBP competition on Kaggle](https://www.kaggle.com/competitions/isic-2024-challenge). The pipeline consists of five main steps: Data Collection, Exploratory Data Analysis (EDA), Data Processing, Model Training, and Model Validation.

## Setup
Install the requirements.txt file in a conda environment. Use jupytext to convert the main.py file to a notebook.
```bash
conda create --name myenv --file requirements.txt
jupytext --to ipynb main.py
```

## Pipeline

### 1. Data Collection
In this step, the necessary data for the project is collected and organized. It’s ensured that all the necessary permissions to use the data are in place and the balance of the data is checked. If it’s imbalanced, more data for the under-represented classes is collected.

### 2. Exploratory Data Analysis (EDA)
EDA helps in understanding the nature of the data. Patterns, correlations, or anomalies in the data are looked for. Specifically, looking for data distribution, quality, correlation analysis, etc..

### 3. Data Processing
The data is preprocessed with techniques including normalization, handling missing values, and encoding categorical variables. For image data, techniques like resizing, denoising, and augmentation.

### 4. Model Training
Since the project involves working with image data, convolutional neural networks (CNNs) are used. Transfer learning is also considered, which can save training time and potentially improve performance.

### 5. Model Validation
Techniques like cross-validation are used to estimate how well the model will generalize to unseen data. The model’s hyperparameters are tuned to find the best combination that improves the validation score.

## Contact
If there are any questions or suggestions, feel free to open an issue or submit a pull request.
