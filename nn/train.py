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

# convert to embeddings

nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

train_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_length])
train_outputs = tf.placeholder(tf.int32, shape=[batch_size, seq_length])

encoder_inputs = [ tf.squeeze(i) for i in tf.split(train_inputs, seq_length, 1) ]
decoder_inputs = [ tf.squeeze(i) for i in tf.split(train_inputs, seq_length, 1) ]

single_cell = tf.contrib.rnn.BasicLSTMCell(cell_size)
cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(num_layers)])

output_projection = None
def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
  return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
      encoder_inputs,
      decoder_inputs,
      cell,
      num_encoder_symbols=vocabulary_size,
      num_decoder_symbols=vocabulary_size,
      embedding_size=size,
      output_projection=output_projection,
      feed_previous=do_decode,
      dtype=dtype)


target_weights = tf.placeholder(dtype, shape=[None], name="target_weights")

targets = [decoder_inputs[i + 1] for i in xrange(len(decoder_inputs) - 1)]
buckets = [(3,3)]
softmax_loss_function = None

self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
    encoder_inputs, decoder_inputs, targets,
    target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
    softmax_loss_function=softmax_loss_function)

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
