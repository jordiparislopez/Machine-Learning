"""
    Image_rec_conv.py

    This code performs an image recognition routine using a convolutional
    neural network. The results shall be similar than the ones obtained in the
    image_rec.py file.

"""


# Import tensorflow as tf
import tensorflow as tf

# Define callback. In this file, we are using the one that stops the training.
class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.99):   # Condition for stopping.
      print("99% accuracy reached. \nStopping routine.\n") # Print message.
      self.model.stop_training = True   # Activation of the break.

# Import mnist dataset from keras and assign it to variables.
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Reshape the training and test x_values into a 4-dimensional array following:
# Array = (tranining_examples , height_size, width_size , colour_layer).
# Training and test exemples are given by the mnist dataset.
training_images=training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# Scale our input to belong to values between 0 and 1
training_images=training_images / 255.0
test_images=test_images/255.0

# Definition of convolutional model:
# 1. Use several (32) 2-dimensional filters (3x3), with relu activation.
# 2. Use MaxPooling to half the input_data.
# 3. Flatten the dataset into a 1-dimensional vector
# 4. Use a hidden layer with relu activation
# 5. Apply a softmax activation on a final layer with the number of output classes
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model using adam, categorical crossentropy and follow accuracy metrics
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Call callbacks and use them to fit the model.
callbacks = myCallback()
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

# Evaluate test values to see how trustworthy is our training.
model.evaluate(test_images, test_labels)
