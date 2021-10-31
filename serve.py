import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

from flask import Flask, request, Response #pip install flask
from flask.json import jsonify
import numpy as np         # dealing with arrays
import os                  # dealing with directories
import sys
import json
from operator import itemgetter
import pandas as pd
from flask_cors import CORS #pip install -U flask-cors

tokenizer = Tokenizer()

MODEL_NAME = 'nlp-tweet'

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

input_sequences = []
for line in all_sentences:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

def get_model_api():
	# make the model
	model = Sequential()
	model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
	model.add(Bidirectional(LSTM(150)))
	model.add(Dense(total_words, activation='softmax'))
	print('LOADING MODEL:', MODEL_NAME)
	if os.path.exists(MODEL_NAME):
		model = keras.models.load_model(MODEL_NAME)
		print(model.summary())

	def model_api(seed_text):
		next_words = 20
		for _ in range(next_words):
			token_list = tokenizer.texts_to_sequences([seed_text])[0]
			token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
			predicted = model.predict(token_list, verbose=0)
			predicted = np.argmax(predicted)
			output_word = ""
			for word, index in tokenizer.word_index.items():
				if index == predicted:
					output_word = word
					break
			seed_text += " " + output_word
		return seed_text
		
	return model_api