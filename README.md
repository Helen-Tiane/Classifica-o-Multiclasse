# Multiclass Classification
Autora: Helen Tiane Alves
# Overview

This notebook presents a multiclass classification implementation using a neural network architecture in PyTorch.
The objective is to classify images of bananas at different stages of ripeness (green, ripe, overripe, and rotten).
The code includes everything from importing libraries to defining the model architecture, including data preprocessing and training.

> Dataset available at: (https://universe.roboflow.com/zd7y6xooqnh7wc160ae7/banana-ripeness-classification/dataset/5)

<img width="1064" height="277" alt="image" src="https://github.com/user-attachments/assets/c075cbad-4d83-44a9-abf2-353755b0d328" />


# Main Components

## 1. Libraries Used

PyTorch: For building and training the neural network.

NumPy: Manipulating numerical arrays.

PIL/Pandas: Image processing and data manipulation.

Matplotlib/Seaborn: Data visualization and confusion matrix.

Scikit-learn: Evaluation metrics such as the confusion matrix.

## 2. Model Architecture

Architecture class: Defines the structure of the neural network, including:

Model initialization, loss function, and optimizer.

Methods for training, validation, prediction, and visualization of results.

Helper functions for calculating metrics and applying hooks for filter visualization.

## 3. Data Preprocessing

Transformations: Resizing, normalization, and conversion to tensors.

Images were resized to: [3, 28, 28]

Dataset: Using ImageFolder to load images of different classes.

Dataset size: 11,793 images

Number of classes: 4

DataLoader: Splitting the data into training and validation sets, with class balancing.

## 5. Training and Evaluation

Loss Function: CrossEntropyLoss for multiclass problems.

Optimizer: Adam, with learning rate adjustment.

Metrics: Loss tracking (training/validation) and confusion matrix for evaluation.

Visualization: Loss graphs and confusion matrix for performance analysis.

<img width="977" height="378" alt="loss" src="https://github.com/user-attachments/assets/58ce774d-afe6-4301-844a-da47c3ddda47" />


<img width="989" height="489" alt="image" src="https://github.com/user-attachments/assets/e2f1843d-c295-41d6-9629-6b5863803c98" />


<img width="886" height="750" alt="image" src="https://github.com/user-attachments/assets/3f6ef5b0-a6f8-4f76-a804-8fe806c577e0" />



## 7. Results

### Training Accuracy Percentage.

Based on the runs, we saw that the arch_cnn2 model (with dropout) correctly matched 9,483 images out of a total of 11,793 images in the training set.
Therefore, the model's accuracy percentage on the training set is approximately 80.41%.

It is important to note that accuracy on the training set is generally higher than on the validation set, as the model saw and learned from this data during training.
In this case, the accuracy on the validation set was approximately 85.13%, which suggests that the model is generalizing well.

We also saw that the arch_cnn2_nodrop model (without dropout) correctly matched 8,944 images out of a total of 11,793 images in the training set.
The model's accuracy percentage on the training set for the model without dropout is approximately 75.84%.

Compared to the dropout model (80.41% training accuracy), the dropout model performed slightly better on the training set in this specific case.

Comparing validation accuracies:

Dropout model: 85.13%

Non-Dropout model: 80.14%


