"""
    Image_class_2.py

    This file uses a dataset of cats and dogs RGB pictures to be
    classified binarily (either cat or dog). The mechanics behind this code
    follow the same as image_class_1, with some slight modifications or
    additions that can be of interest.

"""

# Import tensorflow, os, wget, random and zipfile libraries.
import tensorflow as tf
import os
import wget
import zipfile
import random

# Import additional modules for the optimizer and ImageDataGenerator.
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

# If the URL doesn't work, visit https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
# And right click on the 'Download Manually' link to get a new URL to the dataset.


# Import dataset. It will take time to download.
url ="https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"
wget.download(url, "/tmp/cats-and-dogs.zip")

# Extract zip files
local_zip = '/tmp/cats-and-dogs.zip'
zip_ref   = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

# Create training and testing cats and dogs directories to classify the data.
try:
    os.mkdir('/tmp/cats-v-dogs')
    os.mkdir('/tmp/cats-v-dogs/training')
    os.mkdir('/tmp/cats-v-dogs/testing')
    os.mkdir('/tmp/cats-v-dogs/training/cats')
    os.mkdir('/tmp/cats-v-dogs/training/dogs')
    os.mkdir('/tmp/cats-v-dogs/testing/cats')
    os.mkdir('/tmp/cats-v-dogs/testing/dogs')
except OSError:
    pass


# Definition of split_data function to split dataset into training and testing.
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []                              # Define an empty array.
    for filename in os.listdir(SOURCE):     # Loop over every file.
        file = SOURCE + filename            # Directory of filename.
        if os.path.getsize(file) > 0:       # If file exists,
            files.append(filename)          # append directory to files,
        else:                               # else ignore the file.
            pass

    training_length = int(len(files) * SPLIT_SIZE)      # Define train according to SPLIT_SIZE.
    testing_length = int(len(files) - training_length)  # Define test according to training_length.
    shuffled_set = random.sample(files, len(files))     # Shuffle the files randomly.
    training_set = shuffled_set[0:training_length]      # Get the training set files.
    testing_set = shuffled_set[-testing_length:]        # Get the testing set files.

    for filename in training_set:            # Copy file to training folder.
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:            # Copy file to testing folder.
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)

# Define destination folders.
CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

# Split data according to the split size (95% training, 5% test).
split_size = .95
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)


# Definition of convolutional model.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Sigmoid for 0 or 1 classification.
])


# Compile model with RMSprop and learning rate 0.001.
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

# Manipulation of files within the training directory using ImageDataGenerator.
# The more options there are, the longer the code it takes.
TRAINING_DIR = "/tmp/cats-v-dogs/training/"
train_datagen = ImageDataGenerator(rescale=1./255, # Rescale so values are in (0,1)
      rotation_range=40,                           # Rotate 40º
      width_shift_range=0.2,                       # Generate width-shifted files
      height_shift_range=0.2,                      # Generate height-shifted files
      shear_range=0.2,                             # Shear the picture
      zoom_range=0.2,                              # Zoom the picture
      horizontal_flip=True,                        # Flip the picture horizontally
      fill_mode='nearest')                         # Fill sheared holes

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,         # In 100 batches,
                                                    class_mode='binary',    # binarily,
                                                    target_size=(150, 150)) # leaving 150x150 files.




# Manipulation of files within the testing directory using ImageDataGenerator.
VALIDATION_DIR = "/tmp/cats-v-dogs/testing/"
validation_datagen = ImageDataGenerator(rescale=1./255, # Rescale so values are in (0,1)
      rotation_range=40,                           # Rotate 40º
      width_shift_range=0.2,                       # Generate width-shifted files
      height_shift_range=0.2,                      # Generate height-shifted files
      shear_range=0.2,                             # Shear the picture
      zoom_range=0.2,                              # Zoom the picture
      horizontal_flip=True,                        # Flip the picture horizontally
      fill_mode='nearest')                         # Fill sheared holes
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=100,         # In 100 batches,
                                                              class_mode='binary',    # binarily,
                                                              target_size=(150, 150)) # leaving 150,150 files.

# Fit the model using fit_generator. Save data to history variable.
history = model.fit_generator(train_generator,
                              epochs=50,
                              verbose=1,
                              validation_data=validation_generator)






#-----------------------------------------------------------
#                   PLOTTING THE RESULTS
#-----------------------------------------------------------

"""
    This part of the file focuses on plotting the relevant results using
    modules from matplotlib.
"""

# Import matplotlib image and pyplot modules
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt


# Retrieve values of accuracy and loss of training and validation sets
# From history variable.
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc)) # Get number of epochs

# Plot accuracy in function of epochs of both datasets.
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

# Plot loss in function of epochs of both datasets.
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.figure()
