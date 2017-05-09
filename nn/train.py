# train using this https://www.tensorflow.org/tutorials/word2vec

import tensorflow as tf
import numpy as np
import math
from collections import Counter
import pdb
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib
from tensorflow.contrib.legacy_seq2seq import model_with_buckets

INPUT_FILE_NAME = 'input.txt'
OUTPUT_FILE_NAME = 'output.txt'
 
words = Counter()
def load_words(file_name):
  with open(file_name, 'r') as f:
    for line in f:
      words.update(line.lower().split())

load_words(INPUT_FILE_NAME)
load_words(OUTPUT_FILE_NAME)

word_to_id = {}
GO = 0
EOS = GO+1
PAD = EOS+1
UNK = PAD+1
id_to_word = ["GO", "EOS", "PAD", "UNK"]

id = len(id_to_word)
for word in words.keys():
  word_to_id[word] = id
  id_to_word.append(word)
  assert(id_to_word[id] == word)
  assert(word_to_id[word] == id)
  id += 1

vocabulary_size = len(word_to_id)

print(word_to_id)
print("{0} words".format(vocabulary_size))

# make the input/output values

def load_utterances(file_name, utterances, end_with):
  with open(file_name, 'r') as f:
    for line in f:
      ids = [word_to_id[word] for word in line.lower().split()]
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
size = 1024 
dtype = tf.float32

train_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_length])
train_outputs = tf.placeholder(tf.int32, shape=[batch_size, seq_length])

encoder_inputs = tf.split(tf.cast(train_inputs, tf.float32), seq_length, 1)
decoder_inputs = tf.split(tf.cast(train_inputs, tf.float32), seq_length, 1)

single_cell = tf.contrib.rnn.BasicLSTMCell(cell_size)
cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(num_layers)])

pdb.set_trace()
outputs, states = seq2seq_lib.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)

pdb.set_trace()

'''
outputs, states = seq2seq_lib.embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, vocabulary_size, vocabulary_size, embedding_size, output_projection=None, feed_previous=False)

pdb.set_trace()

num_samples = 512

w_t = tf.get_variable("proj_w", [vocabulary_size, size], dtype=dtype)
w = tf.transpose(w_t)
b = tf.get_variable("proj_b", [vocabulary_size], dtype=dtype)

# inputs: (batch_size, dim)
# labels: (batch_size, num_true)
def sampled_loss(labels, inputs):
  labels = tf.reshape(labels, [-1, 1])
  # We need to compute the sampled_softmax_loss using 32bit floats to
  # avoid numerical instabilities.
  local_w_t = tf.cast(w_t, tf.float32)
  local_b = tf.cast(b, tf.float32)
  local_inputs = tf.cast(inputs, tf.float32)
  return tf.cast(
      tf.nn.sampled_softmax_loss(
          weights=local_w_t,
          biases=local_b,
          labels=labels,
          inputs=local_inputs,
          num_sampled=num_samples,
          num_classes=vocabulary_size),
      dtype)

#outputs = tf.matmul(outputs, 
'''
