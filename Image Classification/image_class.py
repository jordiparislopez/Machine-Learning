"""

    This file uses a dataset of happy and sad 150px x 150px RGB pictures to be
    classified binarily (0 sad, 1 happy). To achieve so, this code must import
    the files, train a model by reducing the size of the picture and apply
    convolutions, and be able to state if the face of the person is happy of sad.

"""

# Import tensorflow, of and zipfile libraries.
import tensorflow as tf
import os
import zipfile

# Import dataset from Laurence Moroney repository
!wget --no-check-certificate \
    "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
    -O "/tmp/happy-or-sad.zip"

# Get the imported zip file and extract the contents into a temporary directory.
zip_ref = zipfile.ZipFile("/tmp/happy-or-sad.zip", 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

# Define callback. In this file, we are using the one that stops the training.
class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.999):   # Condition for stopping.
      print("99.9% accuracy reached. \nStopping routine.\n") # Print message.
      self.model.stop_training = True   # Activation of the break.

# Definition of the convolutional model following:
# 1. Apply a sequence of 2D convolutions (3x3) and maxpooling to reduce input size.
# 2. Flatten the data.
# 3. Apply one final relu activation.
# 4. Apply a sigmoid activation (sending most of the values to 0 or 1)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model using an RMSprop optimizer.
from tensorflow.keras.optimizers import RMSprop       # Import optimizer
model.compile(loss = 'binary_crossentropy',           # Binary loss function
              optimizer = RMSprop(lr = 0.001),        # Optimizer with learning rate
              metrics = ['acc'])                      # Accuracy metric

# Rescale input picture to guarantee that the values belong in the range [0,1]
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255)


# Flow dataset and resize input using flow_from_directory function.
train_generator = train_datagen.flow_from_directory(
        "/tmp/h-or-s",                    # Choose directory
        target_size = (150, 150),         # Resize every picture to 150x150
        batch_size = 10,                  # Flow in different batches
        class_mode = 'binary')            # Use binary crossentropy



# Fit model after calling the callbacks defined and save it to history.
callbacks = myCallback()
history = model.fit_generator(
      train_generator,
      steps_per_epoch = 2,
      epochs = 15,
      verbose = 1,
      callbacks = [callbacks])
