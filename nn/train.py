# train using this https://www.tensorflow.org/tutorials/word2vec

import tensorflow as tf
import numpy as np
import math
from collections import Counter
import pdb
from tf.contrib.legacy_seq2seq import embedding_rnn_seq2seq


INPUT_FILE_NAME = 'input.txt'
OUTPUT_FILE_NAME = 'output.txt'
 
words = Counter()
def load_words(file_name):
  with open(file_name, 'r') as f:
    for line in f:
      words.update(line.lower().split())

load_words(INPUT_FILE_NAME)
load_words(OUTPUT_FILE_NAME)

keys = {}
GO = 1
EOS = GO+1
id = EOS+1
for word in words.keys():
  keys[word] = id
  id += 1

vocabulary_size = len(keys)

print(keys)
print("{0} words".format(vocabulary_size))

# make the input/output values

def load_utterances(file_name, utterances, end_with):
  with open(file_name, 'r') as f:
    for line in f:
      ids = [keys[word] for word in line.lower().split()]
      ids.append(end_with)
      utterances.append(ids)

inputs = []
load_utterances(INPUT_FILE_NAME, inputs, GO)
print("Inputs")
print(inputs)

outputs = []
load_utterances(OUTPUT_FILE_NAME, outputs, EOS)
print("Outputs")
print(outputs)

batch_size = 5
seq_length = 4
embedding_size = 128
cell_size = 96
num_layers = 3

# convert to embeddings

nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

encoder_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_length])
decoder_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_length])

single_cell = tf.contrib.rnn.BasicLSTMCell(cell_size)
cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(num_layers)])

outputs, states = embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, vocabulary_size, vocabulary_size, embedding_size, output_projection=None, feed_previous=False)

