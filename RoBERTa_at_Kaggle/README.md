# README

## Instructions

This project involves training a multi-label text classification model using the Civil Comments dataset. The key steps in the code are:

- **Data Preparation**: Loading the Civil Comments dataset, binarizing the labels based on predefined thresholds, and creating balanced datasets for training, validation, and testing.
- **Model Training**: Using the `RobertaForSequenceClassification` model from Hugging Face Transformers library for multi-label classification.
- **Metrics Calculation**: Evaluating the model's performance with metrics such as accuracy, precision, recall, F1 score, ROC AUC, and additional metrics specific to multi-label tasks.
- **Visualization**: Generating plots for loss curves, ROC curves, and a cross-label confusion matrix to analyze the results.

1. Copy this code to a Kaggle notebook, choose Nvidia P100 GPU as the accelerator, and execute the notebook cells in order.

2. The visualization and metrics will be generated at the end. Each epoch will take approximately 21 minutes, and there are 5 epochs in total.
