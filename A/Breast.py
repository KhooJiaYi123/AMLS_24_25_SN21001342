import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

data = np.load(r"Dataset/breastmnist.npz")

#Label for 0 = malignant , 1 = benign

#Splitting data into X,Y for input features and output category (Train, Validation, Test). 
#normalising data to obtain dataset with feature values between 0 to 1. 
X_train = data["train_images"].astype("float32") / 255
y_train = data["train_labels"]

X_validation = data["val_images"].astype("float32") / 255
y_validation = data["val_labels"]

X_test = data["test_images"].astype("float32") / 255
y_test = data["test_labels"]


#####################################################################################################################################

### Convolution neural network MODEL (Sequential) ###

#####################################################################################################################################


# Hyperparameters Setting 

EPOCHS = 100
BATCH_SIZE = 32
OPTIMIZER = keras.optimizers.Adam(learning_rate=7e-4)

#Model Shape
#Using Keras library to build a sequential, 2 layered CNN --> layer 1: 64 nodes layer 2:128 nodes
#Rectified Liner Unit chosen as activation function
#Sigmoid as activation function for dense layer
model = keras.Sequential()
model.add(keras.layers.Input(shape=[28, 28, 1]))

model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))

model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1, activation="sigmoid"))


# Overfitting issue fixes
#Implemented a restore to obtain the weights for best validation loss before overfitting 
#Patience set to 15 (allows 15 more epochs since lowest val loss measured to check if validation loss will further decrease)
callback = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
#use binary cross entropy/binary accuracy since task is binary classification
model.compile(loss=keras.losses.binary_crossentropy, optimizer=OPTIMIZER, metrics=[keras.metrics.binary_accuracy])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=[X_validation, y_validation], callbacks=callback)



#Plotting Validation accuracy and Training accuracy to assess model performance and observe overfitting
plt.plot(history.history["binary_accuracy"],label="training accuracy")
plt.plot(history.history["val_binary_accuracy"],label="validation accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

#Plotting Validation loss and Training loss to assess model performance and observe overfitting
plt.plot(history.history["loss"],label="training loss")
plt.plot(history.history["val_loss"],label="validation loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()


### Model Prediction and performance on test set###

#Predicted output gives probabilistic values between 0 to 1, hence the values are assigned 1 or 0 based on if the prediction is more than or less than 0.5)
results = ((model.predict(X_test) > 0.5)).astype("uint8")

#Using Metrics library to compute the scores for verification 
accuracy = accuracy_score(y_test, results)


print("=================================================================")
print("Classification Report - Test Set CNN")
print("=================================================================")
# Classification report for detailed metrics
print(classification_report(y_test, results))
print("Validation Accuracy:", accuracy)

#Verified that previous score calculations line up with classification report. (Scores match for 0 section in report - Scores when assuming 0 is the accessed category)

#####################################################################################################################################

### LOGISTIC REGRESSION MODEL ###

#####################################################################################################################################

#Reloading data (personal preference)

data = np.load(r"Dataset/breastmnist.npz")
X_train = data["train_images"]
y_train = data["train_labels"]

X_validation = data["val_images"]
y_validation = data["val_labels"]

X_test = data["test_images"]
y_test = data["test_labels"]

#### FLATTEN FOR LR MODEL####
# Flatten X_training and X_validation
X_train_flat = X_train.reshape(X_train.shape[0], -1) 
X_validation_flat = X_validation.reshape(X_validation.shape[0], -1) 

# Flatten Y_training and Y_validation (Required for LR model from sklearn)
Y_train_flat = y_train.ravel() 
Y_validation_flat = y_validation.ravel()  

#Test set flattening 
X_test_flat = X_test.reshape(X_test.shape[0], -1)
Y_test_flat = y_test.ravel()
# Training the Logistic Regression Model (Setting base iteration) 
model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train_flat, Y_train_flat)

# Predict using the validation set 
Y_pred_LR_Val = model2.predict(X_validation_flat)

# Calculate accuracy using metrics from sklearn 
accuracyLR = accuracy_score(Y_validation_flat, Y_pred_LR_Val)


# Classification report for accuracy scores and performance
print("=================================================================")
print("Classification Report - Validation Set Logistic Regression")
print("=================================================================")
print(classification_report(Y_validation_flat, Y_pred_LR_Val))
print("Validation Accuracy:", accuracyLR)

##HYPERPARAMATER TUNING###

# Define hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strengths to test 
    'solver': ['liblinear', 'lbfgs'],  #Base solvers used
}

# using grid search with cross validation to search for paramter with the best performances 
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid_search.fit(X_train_flat, Y_train_flat)

# Print the best parameters for record
print("Best hyperparameters:", grid_search.best_params_)

#Use best parameter values 
model2 = grid_search.best_estimator_

#Using tuned model on the test set to evaluate performance (Logistic Regression Model)
Y_pred_LR_test = model2.predict(X_test_flat)

print("=================================================================")
print("Classification Report - Test Set Logistic Regression")
print("=================================================================")
print(classification_report(Y_test_flat, Y_pred_LR_test))
print("Test Accuracy:", accuracy_score(Y_test_flat,Y_pred_LR_test))


if __name__ == "__main__":
    main()