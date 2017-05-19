# train using this https://www.tensorflow.org/tutorials/word2vec

import tensorflow as tf
import numpy as np
import pickle
import math
import os
import sys
from collections import Counter
import pdb
import argparse
import re
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib
from tensorflow.contrib.legacy_seq2seq import model_with_buckets

parser = argparse.ArgumentParser(description="Train and sample dialogs")
parser.add_argument('--sample', action='store_true', default=False, help='Sample the current saved model')
parser.add_argument('--epochs', type=int, default=0, help='Run the training with specified number of epochs. The default is zero')
parser.add_argument('--save_words', action='store_true', default=False, help='Save the words file')
args = parser.parse_args()

debug = False
def dprint(v):
  if debug:
    print(v)

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

#INPUT_FILE_NAME = 'input.txt'
#OUTPUT_FILE_NAME = 'output.txt'

INPUT_FILE_NAME = 'test.enc'
OUTPUT_FILE_NAME = 'test.dec'

# remove non-alpha from input to help with density
def clean_line(line):
  return re.sub(r"[^A-Za-z ]+", '', line)

words = Counter()
def load_words(file_name):
  with open(file_name, 'r') as f:
    for line in f:
      line = clean_line(line)
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

#print(word_to_id)
print("{0} words".format(vocabulary_size))

# make the input/output values

def load_utterances(file_name, utterances, end_with):
  with open(file_name, 'r') as f:
    for line in f:
      line = clean_line(line)
      ids = [word_to_id[word] for word in line.lower().split()]
      ids.append(end_with)
      utterances.append(ids)
    return max

data_inputs = []
load_utterances(INPUT_FILE_NAME, data_inputs, PAD)
#print("Inputs max_input({0})".format(max_input))
#print(data_inputs)

data_outputs = []
load_utterances(OUTPUT_FILE_NAME, data_outputs, EOS)
#print("Inputs max_output({0})".format(max_output))
#print("Outputs")
#print(data_outputs)

max = 0
for di in data_inputs:
  if len(di) > max:
    max = len(di)
for di in data_outputs:
  if len(di) > max:
    max = len(di)

print("Max length is {0}".format(max))

batch_size = 10
seq_length = 10
embedding_size = 128
cell_size = 96
num_layers = 3
size = 1024 
dtype = tf.float32

train_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_length], name="train_inputs")
train_outputs = tf.placeholder(tf.int32, shape=[batch_size, seq_length+1], name="train_outputs")

"abc"
# old encoder_inputs = tf.split(tf.cast(train_inputs, tf.float32), seq_length, 1)
encoder_inputs = [ tf.squeeze(i) for i in tf.split(tf.cast(train_inputs, tf.int32), seq_length, 1)]
"<go>abc"
# old decoder_inputs = tf.split(tf.cast(train_outputs, tf.float32), seq_length+1, 1)
decoder_inputs = [ tf.squeeze(i) for i in tf.split(tf.cast(train_outputs, tf.int32), seq_length+1, 1)]

single_cell = tf.contrib.rnn.BasicLSTMCell(cell_size)
cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(num_layers)])

# outputs [(2,6)]*3
outputs, states = seq2seq_lib.embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, vocabulary_size, vocabulary_size, 128)
logits = outputs
probs = [char[0] for char in outputs]
probs = [tf.nn.softmax(char) for char in probs]
probs = [tf.argmax(char, 0) for char in probs]

"abc<eos>"
# targets [2]*2
targets = [decoder_inputs[i + 1] for i in xrange(len(decoder_inputs) - 1)]
# targets [2]*3
targets.append( tf.constant([EOS]*batch_size, dtype=tf.int32) )

loss = seq2seq_lib.sequence_loss(logits, targets, [tf.ones([batch_size]) for _ in range(len(logits))], vocabulary_size)
lr = tf.Variable(0.001, trainable=False)
tvars = tf.trainable_variables()

global_step = tf.Variable(0, trainable=False)
lr_op = tf.train.exponential_decay(0.001, global_step, 1, 0.9999)

optimizer = tf.train.AdamOptimizer(lr_op)

cost_op = tf.reduce_sum(loss) / batch_size / seq_length
use_clipping = False
if use_clipping:
  grads= tf.gradients(cost_op, tvars)
  grad_clip = 5
  tf.clip_by_global_norm(grads, grad_clip)
  grads_and_vars = zip(grads, tvars)
  train_op = optimizer.apply_gradients(grads_and_vars)
else:
  train_op = optimizer.minimize(loss)

tf.get_variable_scope().reuse_variables()

sample_outputs, sample_states = seq2seq_lib.embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, vocabulary_size, vocabulary_size, 128, feed_previous=True)
sample_outputs = [char[0] for char in sample_outputs]
sample_outputs = [tf.nn.softmax(char) for char in sample_outputs]
sample_outputs = [tf.argmax(char, 0) for char in sample_outputs]

session = tf.Session()
session.run(tf.global_variables_initializer())

#training is wrong
#reverse ordering
epochs = args.epochs
stop = False

def get_decoder_value(value, pos):
  if pos < len(value):
    return value[pos]
  if pos == len(value):
    return EOS
  return PAD

def get_encoder_value(value, pos):
  if pos < len(value):
    return value[pos]
  return PAD

saver = tf.train.Saver()
save_dir = "./saves/train.ckpt"
words_file = "./saves/words.txt"
pickle_file = "./saves/pickle_file"

start = 0
if os.path.exists(pickle_file):
  with open(pickle_file, 'rb') as input:
    props = pickle.load(input)
    start = props[ "epoch" ] + 1
  saver.restore(session, save_dir)


for epoch in range(0, epochs):
  if stop:
    break

  i = 0
  while True:

    if i+batch_size > len(data_inputs):
      break

    ti = np.zeros((batch_size, seq_length))
    to = np.zeros((batch_size, seq_length+1))
   
    for r in range(0, batch_size):
      for c in range(0, seq_length):
        ti[r][c] = get_encoder_value(data_inputs[r+i], c)
        #ti[r][c] = data_inputs[r+i][c]

    for r in range(0, batch_size):
      to[r][0] = GO
      for c in range(0, seq_length):
        to[r][c+1] = get_decoder_value(data_outputs[r+i], c)
        #to[r][c+1] = data_outputs[r+i][c]
      

    feed_dict = { train_inputs: ti, train_outputs: to, global_step: epoch }
    cost, o_probs, train, lr, o_encoder_inputs, o_decoder_inputs, o_logits, o_targets, o_loss = session.run([cost_op, probs, train_op, lr_op, encoder_inputs, decoder_inputs, logits, targets, loss], feed_dict)

    print("Epoch {2}, Batch {0}, cost {1}, rate{3}".format(i/5, cost, epoch, lr))
    dprint("o_logits:{0}".format(o_logits))
    dprint("o_targets:{0}".format(o_targets))
    dprint("o_probs:{0}".format(o_probs))
    dprint("o_loss:{0}".format(o_loss))
    dprint("o_encoder_inputs:{0}".format(o_encoder_inputs))
    dprint("o_decoder_inputs:{0}".format(o_decoder_inputs))

    '''
    cost_after = session.run(cost_op, feed_dict)
    if cost_after < cost:
      dprint("COST OKAY")
    else:
      dprint("COST HIGHER")
      stop = True
      break
    '''

    i = i + batch_size

  save_path = saver.save(session, save_dir)

  with open(pickle_file, 'wb') as output:
    epoch = pickle.dump({ "epoch" : epoch }, output)
  print("Saved to {0}".format(save_path))

  #print(id_to_word)
  print("Done training")  

if args.save_words:
  with open(words_file, 'wb') as output:
    for word in id_to_word:
      output.write("{0}\n".format(word))
  
if args.sample:
  print("Testing")
  while(True):
    line = raw_input("Enter test value")

    words = line.lower().split()
    if len(words) == 0:
      break

    ti = np.zeros((batch_size, seq_length))
    to = np.zeros((batch_size, seq_length+1))

    for i in range(seq_length):
      ti[0][i] = PAD
    for i in range(len(words)):
      ti[0][i] = get_id(words[i])

    feed_dict = { train_inputs: ti, train_outputs: to } 
    #indexes, o_targets, o_decoder_inputs = session.run([sample_outputs, targets, decoder_inputs], feed_dict)
    indexes = session.run(sample_outputs, feed_dict)

    #print("o_decoder_inputs:{0}".format(o_decoder_inputs))
    #print("o_targets:{0}".format(o_targets))

    print(indexes) 
    seq = []
    for idx in indexes:
      seq.append(id_to_word[idx]) 
    print(seq)
