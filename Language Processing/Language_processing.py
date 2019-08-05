"""
    Language_processing.py

    This file shows how to train a language-processing system, allowing the
    code to predict words in function of the ones previously used.

"""


# Import csv module to deal with text files.
# Import also wget, numpy and tensorflow.
import csv
import wget
import tensorflow as tf
import numpy as np

# Import tokenizer and pad_sequences to manipulate text to data.
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import text from file.
url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv'
wget.download(url,'/tmp/bbc-text.csv')

# Definition of text parameters
vocab_size = 1000           # Maximum size of different words.
embedding_dim = 16          # Dimensions of the embedding vector.
max_length = 120            # Max length of every text segment.
trunc_type='post'           # Should we truncate, we take the first elements using 'post'
padding_type='post'         # Set the padding type to 'post'
oov_tok = "<OOV>"           # Include out of vocabulary term as <OOV>
training_portion = .8       # Ratio of training data vs validation data.


# Create empty lists of sentences and labels to classify kind of word.
sentences = []
labels = []

#Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


# Routine to read text sentences.
with open("/tmp/bbc-text.csv", 'r') as csvfile:     # Open text file.
    reader = csv.reader(csvfile, delimiter=',')     # Define string delimited by ','.
    next(reader)
    for row in reader:                              # In every row:
        labels.append(row[0])                       # Append row[0] to labels.
        sentence = row[1]                           # The sentence is now row[1].
        for word in stopwords:                      # Replace every stopword for an empty space
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
        sentences.append(sentence)                  # Append sentence.


# Set training and validation parameters
train_size = int(len(sentences) * training_portion) # Size of training dataset.
train_sentences = sentences[:train_size]            # Fetch number of training sentences.
train_labels = labels[:train_size]                  # Fetch number of training labels.
validation_sentences = sentences[train_size:]       # Fetch number of validation sentences.
validation_labels = labels[train_size:]             # Fetch number of validation labels.


# Use tokenizer to quantify text dataset according to the number of words and OoV token.
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index           # Generation of word-to-number dictionary

# Obtain the numerical padded values for both datasets.
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)

# Tokenize labels as well and save them into an np.array.
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

# Define the model using:
# 1. Embedding layer to relate words with the vector
# 2. Pooling layer.
# 3. Relu activation layer.
# 4. Softmax of 6 different classes
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

# Compile model.
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# Fit the model using 30 epochs.
num_epochs = 30
history = model.fit(train_padded,
            training_label_seq,
            epochs=num_epochs,
            validation_data=(validation_padded,
            validation_label_seq),
            verbose=2)




# -------------------------------
#           PLOTTING
# -------------------------------

# Import matplotlib for plotting.
import matplotlib.pyplot as plt

# Definition of our plotting function.
def plot_graphs(history, string, title_string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.savefig(title_string + ".png")

plot_graphs(history, "accuracy", "accuracy")
plot_graphs(history, "loss", "loss")
