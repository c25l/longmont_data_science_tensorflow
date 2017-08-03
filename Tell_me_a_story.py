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
batch_size = 20
num_batches =5000
seq_length = 55
learning_rate = 1e-3

# * Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sample',dest="sample", action="store_true")
args = parser.parse_args()

if args.sample:
    seq_length=1
    batch_size=1
    dropout=0

# * Data Work
encoder = {}
_data = open("shakespeare.txt",'r').read().lower()
usedata = []
## It's not really fair to the network to expect it to wield every symbol.
crud = [[r"[\d$:,&()\]\[|\t']",""],[r"[-\n ]+"," "],[r"[?!;]","."]]
for xx,yy in crud:
        _data =re.sub(xx,yy, _data)
for xx in _data[batch_size*seq_length*20+3:]:
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
if not args.sample:
    # **  Send stuff to tensorboard
    sess =  tf.Session()
    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs")
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
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
        if b % (num_batches//20) == 0:
            saver.save(sess, "save/ckpt", global_step= b)
            output = "".join([decoder[x] for x in np.argmax(thoughts, axis=-1).reshape(-1)])
            print("sample output: \"{}\"".format(output))
            state = sess.run(initial_state)

# * Sampling
else:
    sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU':0}))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state("save")
    saver.restore(sess, ckpt.model_checkpoint_path)
    state = sess.run(initial_state)
    x = np.array([[encoder[x] for x in "northumberland "]])
    trajectory = []
    for xx in range(1000):
        feed = {input_data:x, initial_state:state}
        [prob, state] = sess.run([probs,final_state], feed)
        _prob = np.cumsum(prob[0,0])
        sample = np.searchsorted(_prob, np.random.rand(1))[0]
        x=np.array([[sample]])
        trajectory.append(decoder[sample])
    print("".join(trajectory))
