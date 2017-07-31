#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import tqdm

nodes_per_layer = 60
layers = 2
batch_size = 80
dropout_keep = 0.95
num_batches =1000
num_epochs=50
seq_length = 15
learn_rate = 0.002
decay_rate = 0.05
grad_clip = 5
corpus_symbols=0 ## Gets reset later on by the actual data.
samples = 2000


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
       "\n":" ",
       ",":"."}

for x in junk:
    _data = _data.replace(x,junk[x])

usedata = []
for xx in _data:
    if xx not in encoder:
        encoder[xx] = len(encoder)
    usedata.append(encoder[xx])
usedata = np.array(usedata)

train_batches = [usedata[index:index+seq_length*batch_size].reshape(batch_size, seq_length)
                 for index in range(0,len(usedata)//(seq_length*batch_size))]
answer_batches = [usedata[index:index+seq_length*batch_size].reshape(batch_size, seq_length)
                  for index in range(1,len(usedata)//(seq_length*batch_size))]

print(train_batches[0], "\n", answer_batches[0])
def data_iterator():
    index=0
    while True:
        if (index +1) > len(train_batches):
            index=0
        yield index
        index += 1
decoder = {encoder[x]:x for x in encoder}
corpus_symbols=len(encoder)

# ### Network design

tf.reset_default_graph() #This resets the graph on cell run, otherwise this cell is run-once.
sess =  tf.Session()
#Then we'll build a deep rnn's cells, with dropout
cells = [rnn.GRUCell(nodes_per_layer) for _ in  range(layers)]
dropout = [rnn.DropoutWrapper(cell, dropout_keep) for cell in cells]
cell = rnn.MultiRNNCell(dropout, state_is_tuple=True)


#Define our input and output data
input_data = tf.placeholder(tf.int32, [None, None], name="input_data")
targets = tf.placeholder(tf.int32, [None, None], name = "targets")
initial_state = cell.zero_state(batch_size, tf.float32)
sampling_initial_state = cell.zero_state(1, tf.float32)
lr = tf.get_variable("lr", [],dtype=tf.float32,
                             initializer = tf.constant_initializer(learn_rate),
                             trainable=False)

#First, Let's do an embedding into the network space
embed = tf.get_variable("embed", [corpus_symbols, nodes_per_layer])
inputs = tf.nn.embedding_lookup(embed, input_data)

#dynamic rnn wrapper for cells
gru_outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)


# and with this, let's deduce the outputs as well
preds = tf.layers.dense(gru_outputs, corpus_symbols, name = "recombine", activation = None)
probs = tf.sigmoid(preds)
usetargets = tf.one_hot(targets, corpus_symbols)

# need to define how wrong we are, and what to do about it.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=usetargets))
optimizer = tf.train.RMSPropOptimizer(lr)
gvs = optimizer.compute_gradients(loss)
capped_gvs, _ = tf.clip_by_global_norm([grad for grad, _ in gvs],grad_clip)
train_op = optimizer.apply_gradients(zip(capped_gvs, [var for _, var in gvs]))


def sample():
    state = sess.run(sampling_initial_state)
    nextitem = np.array([encoder[x] for x in["t","h","e", " "]]).reshape(1,-1)
    output = ""
    for _ in range(samples):
        feed = {input_data: nextitem.reshape(1,-1), targets: nextitem.reshape(1,-1)}
        for i, _x in enumerate(sampling_initial_state):
            feed[_x] = state[i]
        [predicted, state] = sess.run([probs, final_state], feed)
        proto = predicted[:,-1,:]
        proto = np.cumsum(proto)
        nextitem = np.searchsorted(proto,np.random.rand(1)*proto[-1])
        output = output+decoder[nextitem[0]]
    return output

# // Training

sess.run(tf.global_variables_initializer())
data = data_iterator()
try:
    epochs = tqdm.trange(num_epochs, postfix={"lr":learn_rate})
    for a in epochs:
        state = sess.run(initial_state)
        sessions = tqdm.trange(num_batches,postfix={"loss":0.})
        for b in sessions:
            index= next(data)
            feed = {input_data: train_batches[index], targets: answer_batches[index]}
            for i, _x in enumerate(initial_state):
                feed[_x] = state[i]
            _, state, lossval= sess.run([train_op, final_state, loss], feed)
            sessions.set_postfix(loss=float(lossval))
        lrval = sess.run(tf.assign(lr, lr*(1-decay_rate)))
        epochs.set_postfix(lr=float(lrval))
        output = sample()
        print("sample output: \"{}\"".format(output))
except KeyboardInterrupt:
    print("sample output: \"{}\"".format(sample()))
