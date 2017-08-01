#!/usr/bin/env python3
# * Imports
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import tqdm
import argparse

# * Defaults
nodes_per_layer = 100
layers = 2
batch_size = 40
num_batches =501
num_epochs=50
seq_length = 75
learn_rate = 0.0005

# * Arguments
parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', type=str, default='train')
args = parser.parse_args()

mode = args.mode
if mode is not "train":
    seq_length=1
    batch_size=1

# * Data Work
encoder = {}
_data = open("shakespeare.txt",'r').read().lower()
usedata = []
for xx in _data:
    if xx not in encoder:
        encoder[xx] = len(encoder)
    usedata.append(encoder[xx])
usedata = np.array(usedata)
decoder = {encoder[x]:x for x in encoder}
corpus_symbols=len(encoder)

# * Network design
sess =  tf.Session()
cells = [rnn.GRUCell(nodes_per_layer,
                     bias_initializer=tf.constant_initializer(-1.0))
         for _ in  range(layers)]
cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

#Define our input and output data
input_data = tf.placeholder(tf.int32, [None, None], name="input_data")
targets = tf.placeholder(tf.int32, [None, None], name = "targets")
inputs = tf.one_hot(input_data,corpus_symbols, axis=-1)
initial_state = cell.zero_state(batch_size, tf.float32)

#dynamic rnn wrapper for cells
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

# and with this, let's deduce the outputs as well
preds = tf.layers.dense(rnn_outputs, corpus_symbols, name = "recombine", activation = None)
probs = tf.sigmoid(preds)

# need to define how wrong we are, and what to do about it.
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds, labels=targets))
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)


# * Tensorboard stuff
tf.summary.histogram('preds', preds)
tf.summary.histogram('loss', loss)


# * Training
if mode is "train":
    # **  Send stuff to tensorboard
    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs")
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    index=-1
    for a in tqdm.trange(num_epochs):
        state = sess.run(initial_state)
        sessions = tqdm.trange(num_batches,postfix={"loss":0.})
        for b in sessions:
            index = (index+1) % (len(usedata)-1-seq_length*batch_size)
            feed = {input_data: usedata[index:index+seq_length*batch_size].reshape(batch_size, seq_length),
                    targets: usedata[1+index:1+index+seq_length*batch_size].reshape(batch_size, seq_length),
                    initial_state:state}
            _, state, lossval, thoughts= sess.run([train_op, final_state, loss, preds], feed)
            sessions.set_postfix(loss=float(lossval))
            if (a * num_batches + b) % 100 == 0:
                saver.save(sess, "save/ckpt",
                           global_step=a * num_batches + b)

        output = "".join([decoder[x] for x in np.argmax(thoughts, axis=-1).reshape(-1)])
        print("sample output: \"{}\"".format(output))
# * Sampling
else:
    sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU':0}))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state("save")
    saver.restore(sess, ckpt.model_checkpoint_path)
    state = sess.run(initial_state)
    x = np.array([[encoder[" "]]])
    trajectory = []
    for xx in range(2000):
        feed = {input_data:x, initial_state:state}
        [prob,state] = sess.run([probs,final_state], feed)
        _prob = np.cumsum(prob[0,0]*prob[0,0])
        sample = np.searchsorted(_prob, np.random.rand(1)*_prob[-1])[0]
        x=np.array([[sample]])
        trajectory.append(decoder[sample])
    print("".join(trajectory))
