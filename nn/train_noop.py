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
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_seq2seq

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

batch_size = 10
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
  from tensorflow.python.tools import freeze_graph
  from tensorflow.python.tools import optimize_for_inference_lib

  # Freeze the graph

  input_saver_def_path = ""
  input_binary = False
  output_node_names = "sample_outputs"
  restore_op_name = "save/restore_all"
  filename_tensor_name = "save/Const:0"
  output_frozen_graph_name = 'saves/frozen_'+MODEL_NAME+'.pb'
  output_optimized_graph_name = '../app/src/main/assets/noop_optimized_'+MODEL_NAME+'.pb'
  clear_devices = True

  freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                            input_binary, checkpoint_path, output_node_names,
                            restore_op_name, filename_tensor_name,
                            output_frozen_graph_name, clear_devices, "")
  

  # Optimize for inference

  input_graph_def = tf.GraphDef()
  with tf.gfile.Open(output_frozen_graph_name, "r") as f:
      data = f.read()
      input_graph_def.ParseFromString(data)

  output_graph_def = optimize_for_inference_lib.optimize_for_inference(
          input_graph_def,
          #["train_inputs", "train_outputs"], # an array of the input node(s)
          ["train_inputs"], # an array of the input node(s)
          ["sample_outputs"], # an array of output nodes
          tf.int32.as_datatype_enum)
          #tf.float32.as_datatype_enum)

  # the optimizer is removing 'embedding_rnn_seq2seq/rnn/embedding_wrapper/embedding/read'
  # which is causing the java code to not be able to load the graph
  output_graph_def = input_graph_def

  # Save the optimized graph

  f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
  f.write(output_graph_def.SerializeToString())

  # tf.train.write_graph(output_graph_def, './', output_optimized_graph_name)  
