import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

epochs = 20
lr = 0.01

tokenizer = Tokenizer()

# Read the csv file
dataset = pd.read_csv('data.csv')

# get all tweets and put it in a list
all_sentences = []
for index, tweet in dataset.iterrows():
	if len(tweet.full_text)>5:
		all_sentences.append(tweet.full_text.lower())

# generate a token index to all the words
tokenizer.fit_on_texts(all_sentences)
total_words = len(tokenizer.word_index) + 1
#print(tokenizer.word_index['the'])

input_sequences = []
for line in all_sentences:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
xs = input_sequences[:,:-1]
labels = input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# make the model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))

# train the model
adam = Adam(lr=lr)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
history = model.fit(xs, ys, epochs=epochs, verbose=1)
print(model.summary())
print(model)

# save the model in a file
model.save('nlp-tweet')

# method to plot the accuracy
def plot_graphs(history, string):
	plt.plot(history.history[string])
	plt.xlabel('Epochs')
	plt.ylabel(string)
	plt.show()
# call the plot_graphs method
plot_graphs(history, 'accuracy')