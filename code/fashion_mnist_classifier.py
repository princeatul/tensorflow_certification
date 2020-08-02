# required packages
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# using fashion mnist data from the keras dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# checking the shape of training set and testing set
X_train_full.shape
X_train_full.dtype

# creating validation set from the training set and scaling down the input to the range 0-1
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_test = X_test / 255.0

# defining class names for the labels
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker",
               "Bag", "Ankle boot"]

# checking class of an instance of y_train
class_names[y_train[0]]

# defining the model using sequential API
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# checking model summary
model.summary

# checking different model layers
model.layers
hidden1 = model.layers[1]
hidden1.name
model.get_layer('dense') is hidden1

# we can check initiated weights and biases of the hidden layers
weights, biases = hidden1.get_weights()

weights.shape

biases.shape

# setting up model compilation rule. It defines which error to use, which optimizer to use and metrics to calculate
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# fitting the model which means it is training the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

# we can check different properties of history
history.params
history.history

# we can draw accuracy and loss for each epochs
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
# plt.show()
plt.savefig("test.png")

# evaluating model performance on the test set. It estimates the generalisation error
model.evaluate(X_test, y_test)

# using this model to make prediction. As we don't have a new set, I am using training set instances to predict
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = model.predict_classes(X_new)
y_pred
np.array(class_names)[y_pred]

# -------------------------- End of File------------------------------------------------
