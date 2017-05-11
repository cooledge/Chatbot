# train using this https://www.tensorflow.org/tutorials/word2vec

import tensorflow as tf
import numpy as np
import math
from collections import Counter
import pdb
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib
from tensorflow.contrib.legacy_seq2seq import model_with_buckets

def feed_previous_loop(prev, i):
  return prev

# copied so I can add an arg to loop or not
# seq2seq_lib.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)
def basic_rnn_seq2seq(encoder_inputs,
                      decoder_inputs,
                      cell,
                      feed_previous = False,
                      dtype=tf.float32,
                      scope=None):
  with tf.variable_scope(scope or "basic_rnn_seq2seq"):
    _, enc_state = tf.contrib.rnn.static_rnn(cell, encoder_inputs, dtype=dtype)
    lf = None
    if feed_previous:
      lf = feed_previous_loop
    return seq2seq_lib.rnn_decoder(decoder_inputs, enc_state, cell, loop_function=lf)

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

def get_id(word):
  if word in word_to_id.keys():
    return word_to_id[word]
  else:
    return UNK

id = len(id_to_word)
for word in words.keys():
  word_to_id[word] = id
  id_to_word.append(word)
  assert(id_to_word[id] == word)
  assert(word_to_id[word] == id)
  id += 1

vocabulary_size = len(id_to_word)

print(word_to_id)
print("{0} words".format(vocabulary_size))

# make the input/output values

def load_utterances(file_name, utterances, end_with):
  with open(file_name, 'r') as f:
    for line in f:
      ids = [word_to_id[word] for word in line.lower().split()]
      ids.append(end_with)
      utterances.append(ids)

data_inputs = []
load_utterances(INPUT_FILE_NAME, data_inputs, GO)
print("Inputs")
print(data_inputs)

data_outputs = []
load_utterances(OUTPUT_FILE_NAME, data_outputs, EOS)
print("Outputs")
print(data_outputs)

batch_size = 2
seq_length = 4
embedding_size = 128
cell_size = 96
num_layers = 3
size = 1024 
dtype = tf.float32

train_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_length], name="train_inputs")
train_outputs = tf.placeholder(tf.int32, shape=[batch_size, seq_length+1], name="train_outputs")

"abc"
encoder_inputs = tf.split(tf.cast(train_inputs, tf.float32), seq_length, 1)
"<go>abc"
decoder_inputs = tf.split(tf.cast(train_outputs, tf.float32), seq_length+1, 1)

W = tf.get_variable("W", shape=(cell_size, vocabulary_size))
b = tf.get_variable("b", shape=(vocabulary_size))

single_cell = tf.contrib.rnn.BasicLSTMCell(cell_size)
cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(num_layers)])

#outputs, states = seq2seq_lib.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)
outputs, states = basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)
# <tf.Tensor 'concat_1:0' shape=(5, 384) dtype=float32>
outputs = tf.concat(outputs, 1)
outputs = tf.reshape(outputs, [-1, cell_size])
logits = tf.matmul(outputs, W) + b

# setup the var's for sampling
sample_outputs, sample_states = basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, scope="sample_scope")
sample_outputs = tf.concat(sample_outputs, 1)
sample_outputs = tf.reshape(sample_outputs, [-1, cell_size])
sample_logits = tf.matmul(sample_outputs, W) + b
sample_probs = tf.nn.softmax(sample_logits, -1, name='probs')

"abc<eos>"
targets = [decoder_inputs[i + 1] for i in xrange(len(decoder_inputs) - 1)]
targets.append( tf.constant([[EOS]]*batch_size, dtype=tf.float32) )
targets = tf.cast(tf.concat(targets, 1), tf.int32)

loss = seq2seq_lib.sequence_loss_by_example([logits], [targets], [tf.ones([batch_size * (seq_length+1)])], vocabulary_size)
lr = tf.Variable(0.0, trainable=False)
tvars = tf.trainable_variables()

optimizer = tf.train.AdamOptimizer(lr)
cost_op = tf.reduce_sum(loss) / batch_size / seq_length
grads= tf.gradients(cost_op, tvars)
grad_clip = 5
tf.clip_by_global_norm(grads, grad_clip)
grads_and_vars = zip(grads, tvars)
train_op = optimizer.apply_gradients(grads_and_vars)

session = tf.Session()
session.run(tf.global_variables_initializer())

epochs = 200
for epoch in range(epochs):
  i = 0
  while True:

    if i+batch_size > len(data_inputs):
      break

    ti = np.zeros((batch_size, seq_length))
    to = np.zeros((batch_size, seq_length+1))
   
    for r in range(0, batch_size):
      for c in range(0, seq_length):
        if c >= len(data_inputs[r+i]):
          ti[r][c] = PAD
        else:
          ti[r][c] = data_inputs[r+i][c]

    for r in range(0, batch_size):
      to[r][0] = GO
      for c in range(0, seq_length):
        if c >= len(data_outputs[r+i]):
          to[r][c+1] = PAD
        else:
          to[r][c+1] = data_outputs[r+i][c]

    feed_dict = { train_inputs: ti, train_outputs: to }
    
    cost, train = session.run([cost_op, train_op], feed_dict)

    print("Epoch {2}, Batch {0}, cost {1}".format(i/5, cost, epoch))

    i = i + batch_size

print("Done training")  

print("Testing")

while(True):
  line = raw_input("Enter test value")

  words = line.lower().split()
  if len(words) == 0:
    break

  ti = np.zeros((batch_size, seq_length))
  to = np.zeros((batch_size, seq_length+1))

  for i in range(len(words)):
    ti[0][i] = get_id(words[i])

  feed_dict = { train_inputs: ti, train_outputs: to } 
  probs = session.run([sample_probs], feed_dict)

  print(probs)

