import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
#Load in data
data = np.load(r"Dataset/bloodmnist.npz")

# "label": {
#             "0": "basophil",
#             "1": "eosinophil",
#             "2": "erythroblast",
#             "3": "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
#             "4": "lymphocyte",
#             "5": "monocyte",
#             "6": "neutrophil",
#             "7": "platelet",
#         }

#Splitting data into X,Y for input features and output category (Train, Validation, Test). 
#normalising data to obtain dataset with feature values between 0 to 1. 
X_train = data["train_images"].astype("float32") / 255
y_train = data["train_labels"]

X_validation = data["val_images"].astype("float32") / 255
y_validation = data["val_labels"]

X_test = data["test_images"].astype("float32") / 255
y_test = data["test_labels"]


# Using one hot encoding to categorise 8 unique blood type outputs
# Reshape y_train(Needed for sklearn encoder)
y_train_reshaped = y_train.reshape(-1, 1)

# Initialize and fit the OneHotEncoder
encoder = OneHotEncoder(categories='auto', sparse_output=False)
y_train_one_hot = encoder.fit_transform(y_train_reshaped)

#same process for y_validation and y_test data
y_validation_reshaped = y_validation.reshape(-1, 1)
y_validation_one_hot = encoder.fit_transform(y_validation_reshaped)
y_test_reshaped = y_test.reshape(-1, 1)
y_test_one_hot = encoder.fit_transform(y_test_reshaped)

#####################################################################################################################################

### Convolution neural network MODEL (Sequential) ###

#####################################################################################################################################


# Hyperparameters Setting 
EPOCHS = 100
BATCH_SIZE = 32
OPTIMIZER = keras.optimizers.Adam(learning_rate=1e-3)

#Model Shape
#Using Keras library to build a sequential, 2 layered CNN --> layer 1: 64 nodes layer 2:128 nodes
#Rectified Liner Unit chosen as activation function
#softmax as activation function for dense layer
model = keras.Sequential()
model.add(keras.layers.Input(shape=[28, 28, 3]))

model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))

model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(8, activation="softmax"))


# Overfitting issue fixes
#Implemented a restore to obtain the weights for best validation loss before overfitting 
#Patience set to 15 (allows 15 more epochs since lowest val loss measured to check if validation loss will further decrease)
callback = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
#Use categorical cross entropy for multilabel classification task
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=OPTIMIZER, metrics=["accuracy"])
history = model.fit(X_train, y_train_one_hot, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=[X_validation, y_validation_one_hot], callbacks=callback)


#Plotting Validation loss and Training loss to assess model performance and observe overfitting
plt.plot(history.history["val_loss"],label="validation loss")
plt.plot(history.history["loss"],label="training loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()


#Plotting Validation accuracy and Training accuracy to assess model performance and observe overfitting
plt.plot(history.history["val_accuracy"],label="validation accuracy")
plt.plot(history.history["accuracy"],label="training accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()


#Conversion of max value index into integers
y_pred = np.argmax(model.predict(X_test), axis=1)


#Reshaping to match predicted output dimensions
y_true = y_test.reshape((3421,))

#assess performance for scores for individual categories 
accuracy = accuracy_score(y_true, y_pred)


# Compute performance metrics from sklearn library
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')  # 'weighted' accounts for class imbalance
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

print("======================================")
print("Classification Report - CNN")
print("======================================")

# Classification report 
print(classification_report(y_true, y_pred))
print("Validation Accuracy:", accuracy)

############################################################
# K Nearest Neighbour Algorithm
############################################################

#Reloading Data (Personal preferance)
data = np.load(r"Dataset/bloodmnist.npz")
X_train = data["train_images"].astype("float32") / 255
y_train = data["train_labels"]

X_validation = data["val_images"].astype("float32") / 255
y_validation = data["val_labels"]

X_test = data["test_images"].astype("float32") / 255
y_test = data["test_labels"]


# flatten input for KNN algorithm
X_train_flat = X_train.reshape(X_train.shape[0], -1)  
X_validation_flat = X_validation.reshape(X_validation.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

#flattening outputs
Y_train_flat = y_train.ravel()
Y_validation_flat = y_validation.ravel()
Y_test_flat = y_test.ravel()

# Initialize KNN with a chosen number of neighbors
knn = KNeighborsClassifier(n_neighbors=58)

# Train the KNN model
knn.fit(X_train_flat, Y_train_flat)


## Hyperparameter tuning ##

# Define parameter grid to search for best K number
param_grid = {'n_neighbors': [14,15,16]}

# grid search for best perfomance K number
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_flat, Y_train_flat)

# Best K parameter:
print("Best hyperparameters:", grid_search.best_params_)

# Replace with the best estimator in KNN model
knn = grid_search.best_estimator_

#Using validation set to see validation accuracy and performance of tuned model
Y_validation_pred = knn.predict(X_validation_flat)
# Validation performance


print("======================================")
print("Classification Report - (Validation) KNN")
print("======================================")
print(classification_report(Y_validation_flat, Y_validation_pred))
print("Validation Accuracy:", accuracy_score(Y_validation_flat, Y_validation_pred))


#Predict test set using best tuned model
Y_test_pred = knn.predict(X_test_flat)

# Test dataset performance using tuned model


print("======================================")
print("Classification Report - (Test) KNN")
print("======================================")
print(classification_report(Y_test_flat, Y_test_pred))
print("Test Accuracy:", accuracy_score(Y_test_flat, Y_test_pred))





#obtain scores to higher decimal places
accuracy = accuracy_score(Y_test_flat,  Y_test_pred)
precision = precision_score(Y_test_flat,  Y_test_pred, average='weighted')  # 'weighted' accounts for class imbalance
recall = recall_score(Y_test_flat,  Y_test_pred, average='weighted')
f1 = f1_score(Y_test_flat,  Y_test_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

if __name__ == "__main__":
    main()
