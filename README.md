# AMLS_24_25_SN21001342
## ELEC0134 - MLS Coursework 2025 

Assumed that Dataset folder contains breastmnist.npz and bloodmnist.npz



Brief Description:

Task A is a binary classification task. Models used are CNN built using Keras (Tensorflow) and Logistic Regression from SKLearn. The script in Task A is named Breast.py, it outputs learning curve (accuracy) and validation loss graphs for cnn and LR models after training the models. For CNN, a classification report for final testing is included. For LR, two classification reports: one for validation Gridsearch, one for final testing.

Task B is a multilabel classification task. Models used are CNN built using Keras (Tensorflow) and KNN from SKLearn. The script in Task A is named Blood.py, it outputs learning curve (accuracy) and validation loss graphs for cnn and knn models after training the models. For CNN, a classification report for final testing is included. For KNN, two classification reports: one for validation Gridsearch, one for final testing.

Role of Each File:

A contains all code relating to Task A
B contains all code relating to Task B
Dataset is an empty file, assumes that breastmnist and bloodnmist npz files will be inserted. 
main file to run all

Python Packages required to run all:
- numpy
- sklearn
- tensorflow
- keras
- pandas
- matplotlib


**Note: Both scripts contain training processes and hyperparamter tuning using Gridsearch, so the script may take long to run.**

### How to Run
Under AMLS_24_25_SN21001342 directory, run main.py

`python main.py`