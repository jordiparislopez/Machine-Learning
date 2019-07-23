"""

    File that computes a linear regression using Tensorflow given a simple
    set of data.

    In a linear regression problem, we are given an initial set of data,
    which we use to fit to a function y = A·x + B in a layer, and then we
    shall use this fit to predict the y values of particular x.

"""

# Import tenforflow and keras libraries to compute regression.
# Import numpy to create arrays. Import matplotlib for visualitsation.
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Definition of the model using Sequential (initilisation of the NN).
# We use Dense to introduce a layer. Since it is a unique scalar function,
# it is enough to use a layer of one unit.
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# We compile the model using a stochastic gradient descent a mean_squared_error
# loss function.
model.compile(optimizer='sgd', loss='mean_squared_error')

# We generate the arrays of x and y to fit in our model.
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype = float)
ys = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0], dtype = float)

# We fit the model given our data using model.fit.
num_epochs = 3000
history = model.fit(xs, ys, epochs = num_epochs, verbose = 1)

# This command prints the expected value given a value of x.
# Even though one can analiticaly compute this value, tensorflow uses simply
# the numerical inputs to predict the value.
print(model.predict([10.0]))


# We save the values of loss and number of epochs.
training_loss = history.history['loss']
epoch_count = range(1, num_epochs + 1)

# We print in an external png file the values obtained.
plt.plot(epoch_count, training_loss, 'r--')
plt.legend(['Training Loss'])
plt.xlabel('Number Epochs')
plt.ylabel('Loss')
plt.savefig("Loss.png")
