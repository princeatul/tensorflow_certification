# Working on sequential api


# loading the required packages
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

# Loading the required dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# creating a validation set
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# standardizing the input
X_train = X_train / 255.0
X_valid = X_valid / 255.0
X_test = X_test / 255.0

# creating a class list
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# creating a sequential model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))  # X.reshape(-1, 1)
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# checking summary of the model
model.summary()

# plotting model
keras.utils.plot_model(model)

# getting model list of layers
model.layers

# accessing first hidden layers
hidden1 = model.layers[1]

# getting all the weights and biases initialized for the first layer
weights, biases = hidden1.get_weights()

weights

weights.shape
biases.shape

# compiling the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# training the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

# plotting training parameters
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
# plt.show()
plt.savefig("graphs/image_classifier.png")

model.evaluate(X_test, y_test)

# making prediction
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = model.predict_classes(X_new)
y_pred

# saving the model
model.save("model/image_classifier_mnist_11_aug.h5")

# for loading the mode
model = keras.models.load_model("model/image_classifier_mnist_11_aug.h5")

# change the model training process so that it saves the best model based on validation set performance
checkpoint_cb = keras.callbacks.ModelCheckpoint("model/best_model.h5", save_best_only=True)
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb])
model = keras.models.load_model("model/best_model.h5")  # roll back to best model

# defining early stopping callback to interrupt when there no progress on the validation set for a number of epochs
# and it will optionally rollback to the best model
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
model = model.fit(X_train, y_train, epochs=500, validation_data=(X_valid, y_valid),
                  callbacks=[checkpoint_cb, early_stopping_cb])


# creating our own callbacks
class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))


# creating a callback class for stopping the training when accuracy increases  85%
class Mycallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if logs['accuracy'] > 0.6:
            print("\nReached Accuracy so stopping")
            self.model.stop_training = True


# creating a instance of the class
my_calback_object = PrintValTrainRatioCallback()
my_accuracy_callback = Mycallback()

model = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),
                  callbacks=[checkpoint_cb, early_stopping_cb, my_calback_object, my_accuracy_callback])


# ---------------------- EOF ----------------------------

