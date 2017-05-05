# train using this https://www.tensorflow.org/tutorials/word2vec

import tensorflow as tf
import numpy as np
import math
from collections import Counter
import pdb


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

BATCH_SIZE = 5
MAX_LENGTH = 4
embedding_size = 128

# convert to embeddings

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
