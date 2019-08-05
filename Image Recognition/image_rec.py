"""
    Image_rec.py

    This file contains the routine to develop an image recognition algorithm
    using Tensorflow and Keras. Here, we use the open source mnist
    dataset, that includes a large amount of 28px x 28px grayscale pictures of
    10 diferent classes of objects.

    This code has two purposes. First and foremost, this codes provides a
    routine to recognise and classify pictures after having trained the dataset.
    To achieve so, we use crossentropy loss function model and the softmax
    activation, which states which of the classes have more probability for the
    objects to belong to.

    Secondly, in this code we want to learn about the importance and power of the
    callbacks. We will provide two different callbacks: one that makes the code
    stop once the training achieves certain value of the metrics (accuracy in
    this case), and another one that plays with the terminal output information,
    which allows as to select and read the most relevant data in a more efficient
    way.

"""

# Import tensorflow library as tf and the libraries sys and os
import tensorflow as tf
import sys, os


# Definition of myCallback, which will stop the code when the training
# reaches certain value of accuracy (e.g. 99%).
class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.99):   # Condition for stopping.
      print("99% accuracy reached. \nStopping routine.\n") # Print message.
      self.model.stop_training = True   # Activation of the break.


# Define disable and enable print functions for second callback
def blockPrint():
  sys.stdout = open(os.devnull, 'w')
def enablePrint():
  sys.stdout = sys.__stdout__


# Definition of myCallback1, which will only print even epochs).
class myCallback1(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs={}):
        if(epoch % 2):
            enablePrint()
        else:
            blockPrint() # Block print if odd epoch.

    def on_epoch_end(self, epoch, logs = {}):
        pass


# Import mnist dataset and save it to mnist variable
mnist = tf.keras.datasets.mnist

# Split the data in training and test sets and normalising so that their values
# belong to 0 and 1 according to the grayscale values (in range from 0 to 255).
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# Define our model by:
# 1: Flattening the 28x28 matrices into (28·28,1) vectors:
# 2: Use a hidden layer and apply ReLU (assign x when x > 0, 0 otherwise)
# 3: Apply softmax function to get highest probabilities.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),        # Flattening
  tf.keras.layers.Dense(512, activation=tf.nn.relu),    # First hidden layer
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)   # Output layer
])

# Compile model using the adam optimiser.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Rename our Callback functions.
callbacks = myCallback()
callbacks1 = myCallback1()

# Fit our model to our data and use callbacks to modify output.
# Use callbacks or callbacks1 for different functions.
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

# Evaluate test values to see how trustworthy is our training.
model.evaluate(x_test, y_test)
