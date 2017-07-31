#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import tensorflow.contrib.legacy_seq2seq as seq2seq
import tqdm

nodes_per_layer = 100
layers = 2
dropout_keep_prob = 0.99
batch_size = 100
num_batches =100
num_epochs=101
seq_length = 250
corpus_symbols=39
samples = 500


# ### Data Work

encoder = {}
decoder = {}
_data = open("shakespeare.txt",'r').read().lower()
junk = {";":",",
       "-":" ",
       "3":"",
       "'":"",
        "$":"",
        "&":"",
        "!":".",
       ":":"",
       "\n":"",
       ",":"."}

for x in junk:
    _data = _data.replace(x,junk[x])

for xx in _data:
    if xx not in encoder:
        encoder[xx] = len(encoder)

def data_iterator(slen, batches):
    index=0
    while True:
        if (index + slen+1) > len(_data):
            index=0
        temp = [encoder[x] for x in _data[index:index+slen*batches+1]]
        index += slen
        yield (np.array(temp[:-1]).reshape(batches,slen),
               np.array(temp[1:]).reshape(batches,slen))
decoder = {encoder[x]:x for x in encoder}
corpus_symbols=len(encoder)


# ### Network design

tf.reset_default_graph() #This resets the graph on cell run, otherwise this cell is run-once.
sess =  tf.Session()
#Then we'll build a deep rnn's cells, with dropout
cells = [rnn.GRUCell(nodes_per_layer) for _ in  range(layers)]
dropout_cells = [rnn.DropoutWrapper(cell,dropout_keep_prob) for cell in cells]
cell = rnn.MultiRNNCell(dropout_cells, state_is_tuple=True)


#Define our input and output data
input_data = tf.placeholder(tf.int32, [None, None], name="input_data")
targets = tf.placeholder(tf.int32, [None, None], name = "targets")
initial_state = cell.zero_state(batch_size, tf.float32)
sampling_initial_state = cell.zero_state(1, tf.float32)

#First, Let's do an embedding into the network space
embed = tf.get_variable("embed", [corpus_symbols, nodes_per_layer])
inputs = tf.nn.embedding_lookup(embed, input_data)

#dynamic rnn wrapper for cells
gru_outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)


# and with this, let's deduce the outputs as well
preds = tf.layers.dense(gru_outputs, corpus_symbols, name = "recombine", activation = tf.sigmoid)
usetargets = tf.one_hot(targets, corpus_symbols)

# need to define how wrong we are, and what to do about it.
loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=usetargets)
optimizer = tf.train.AdagradOptimizer(0.02)
train_op = optimizer.minimize(loss)


# // Training

sess.run(tf.global_variables_initializer())
data = data_iterator(seq_length, batch_size)
for a in tqdm.tqdm(range(num_epochs)):
    state = sess.run(initial_state)
    for b in tqdm.tqdm(range(num_batches),leave=0):
        x,y = next(data)
        feed = {input_data: x, targets: y}
        for i, _x in enumerate(initial_state):
            feed[_x] = state[i]
        _, state = sess.run([train_op, final_state], feed)
    state = sess.run(sampling_initial_state)
    nextitem = np.array([encoder[x] for x in["t","h","e", " "]]).reshape(1,-1)
    output = ""
    for _ in range(samples):
        feed = {input_data: nextitem.reshape(1,-1), targets: nextitem.reshape(1,-1)}
        for i, _x in enumerate(sampling_initial_state):
            feed[_x] = state[i]
        [predicted, state] = sess.run([preds, final_state], feed)
        proto = np.cumsum(predicted[:,-1,:])
        nextitem = np.searchsorted(proto,np.random.rand(1)*proto[-1])
        nextitem = np.array(nextitem)
        output = output+ decoder[nextitem[0]]
    print("sample output: \"{}\"".format(output))
