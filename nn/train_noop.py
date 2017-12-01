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
from tensorflow.python.framework import graph_util

parser = argparse.ArgumentParser(description="Train and sample dialogs")
parser.add_argument('--sample', action='store_true', default=False, help='Sample the current saved model')
parser.add_argument('--epochs', type=int, default=0, help='Run the training with specified number of epochs. The default is zero')
parser.add_argument('--save_words', action='store_true', default=False, help='Save the words file')
parser.add_argument('--freeze', action='store_true', default=False, help='Freeze the current graph')
args = parser.parse_args()

debug = False
def dprint(v):
  if debug:
    print(v)

batch_size = 1
seq_length = 20
train_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_length], name="train_inputs")
train_outputs = tf.placeholder(tf.int32, shape=[batch_size, seq_length+1], name="train_outputs")
w = tf.get_variable("w", shape=(seq_length, 10), dtype=tf.int32)
sample_outputs = tf.matmul(train_inputs, w, name="sample_outputs")

session = tf.Session()
session.run(tf.global_variables_initializer())

saver = tf.train.Saver()
MODEL_NAME = 'tfdroid'
checkpoint_path = "./saves/{0}.ckpt".format(MODEL_NAME)
input_graph_path = "./saves/{0}.pbtxt".format(MODEL_NAME)
words_file = "../app/src/main/assets/words.txt"
pickle_file = "./saves/pickle_file"

tf.train.write_graph(session.graph_def, '.', input_graph_path)
save_path = saver.save(session, checkpoint_path)

if args.save_words:
  with open(words_file, 'wb') as output:
    output.write("banana\n")
 
if args.freeze:

  def freeze_it():
    output_file = '../app/src/main/assets/noop_optimized_'+MODEL_NAME+'.pb'
    frozen_graph_def = graph_util.convert_variables_to_constants(session, session.graph_def, ['train_inputs', 'sample_outputs'])
    tf.train.write_graph(
      frozen_graph_def,
      os.path.dirname(output_file),
      os.path.basename(output_file),
      as_text=False)

  freeze_it()
