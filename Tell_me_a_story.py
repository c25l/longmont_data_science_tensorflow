#!/usr/bin/env python3
# * Imports
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import tqdm
import argparse
import re

# * Defaults
nodes_per_layer = 200
layers = 2
batch_size = 60
num_batches =10000
seq_length = 55
out_length = 200
learning_rate = 1e-3

# * Data Work
encoder = {}
_data = open("shakespeare.txt",'r').read().lower()
usedata = []
## It's not as fun to watch the network try to wield every symbol.
crud = [[r"[\d$:,&()\]\[|']",""],[r"[-\t ]+"," "],[r"[?!;]","."]]
for xx,yy in crud:
        _data =re.sub(xx,yy, _data)
for xx in _data:
    if xx not in encoder:
        encoder[xx] = len(encoder)
    usedata.append(encoder[xx])
usedata = np.array(usedata)
decoder = {encoder[x]:x for x in encoder}
corpus_symbols=len(encoder)

# * Network design
cells = [rnn.BasicLSTMCell(nodes_per_layer) for _ in  range(layers)]
cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

#Define our input and output data
input_data = tf.placeholder(tf.int32, [None, None], name="input_data")
targets = tf.placeholder(tf.int32, [None, None], name = "targets")
#Initial state defines the rnn's memory.
initial_state = cell.zero_state(batch_size, tf.float32)

embed = tf.get_variable("embed", [corpus_symbols,nodes_per_layer])
inputs = tf.nn.embedding_lookup(embed, input_data)

#dynamic rnn wrapper for cells
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

# and with this, let's deduce the outputs as well using, what else, more neural network.
preds = tf.layers.dense(rnn_outputs, corpus_symbols, name = "recombine", activation = None)
probs = tf.nn.softmax(preds)

# need to define how wrong we are, and what to do about it.
# This uses softmax cross entropy which is ideal for non-independent outputs.
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds, labels=targets))
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

# * Tensorboard stuff
tf.summary.histogram('preds', preds)
tf.summary.scalar('loss', loss)


# * Training
# **  Send stuff to tensorboard
with tf.Session() as sess:
    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs")
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    state = sess.run(initial_state)
    sessions = tqdm.trange(num_batches,postfix={"loss":0.})
    for b in sessions:
        index = b % (len(usedata)-1-seq_length*batch_size)
        feed = {input_data: usedata[index:index+seq_length*batch_size].reshape(batch_size, seq_length),
                targets: usedata[1+index:1+index+seq_length*batch_size].reshape(batch_size, seq_length),
                initial_state:state}
        _, summary, state, lossval, thoughts= sess.run([train_op, summaries, final_state, loss, preds], feed)
        sessions.set_postfix(loss=float(lossval))
        writer.add_summary(summary, b)
        if (b % (num_batches//10) == 0) or (b == num_batches-1):
            output = "".join([decoder[x] for x in np.argmax(thoughts, axis=-1).reshape(-1)[:out_length]])
            print("sample output : \"{}\"\n".format(output))
            x = np.tile(encoder[" "], (batch_size,1))
            trajectory = []
            usesample = -1
            for xx in range(out_length):
                feed={input_data:x, initial_state:state}
                prob, state = sess.run([probs,final_state], feed)
                sample = np.argmax(prob[0,0])
                if ((sample == encoder[" "]) or (sample==encoder["\n"])
                    or (usesample>0)or (sample==usesample)):
                    usesample=-1
                    sample = np.random.choice(np.arange(prob.shape[-1]), p=prob[0,0])
                    if (sample == encoder[" "]) or (sample == encoder["\n"]):
                        usesample=sample
                x=np.tile(sample, (batch_size,1))
                trajectory.append(decoder[sample])
            print("autonomous output: \"{}\"\n".format("".join(trajectory)))
            state = sess.run(initial_state)
