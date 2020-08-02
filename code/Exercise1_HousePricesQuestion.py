# importing required packages
import tensorflow as tf
import numpy as np
from tensorflow import keras


# Graded function: house_model
def house_model(y_new):
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    ys = np.array([50, 100, 150, 200, 250, 300], dtype=float)
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=500)
    return model.predict(y_new)[0]


prediction = house_model([7.0])[0]
print(prediction)

