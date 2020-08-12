# installing required packages
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


# fetching housing datasets
housing = fetch_california_housing()


# creating different dataset from the housing dataset i.e. training, testing, and validation set
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)


# data preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# defining the architecture of the MLP
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

model.compile(loss="mean_squared_error", optimizer="sgd")


# fitting the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]

y_pred = model.predict(X_new)

print(y_pred)
y_train[:3]


# wide and deep neural network
# created an input object. It specifies the kind of input the model will get
input_ = keras.layers.Input(shape=X_train.shape[1:])
# created a dense layer with 30 neurons. We are calling it like a function and passing the input object.
# That is why this is called the Functional API.  We are defining how keras should connect the layers
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activatio="relu")(hidden1)

concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)

model = keras.Model(inputs=[input_], outputs=[output])


