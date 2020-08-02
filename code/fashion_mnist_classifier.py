# required packages
import tensorflow as tf
from tensorflow import keras


# using fashion mnist data
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_train_full.shape
X_train_full.dtype


X_train_full[0]


# creating validation set from the training set
X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_test = X_test/255.0

# defining class names for the labels
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker",
               "Bag", "Ankle boot"]

class_names[y_train[0]]

# creating the model using sequential API
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
mode.add(keras.layers.Dense)