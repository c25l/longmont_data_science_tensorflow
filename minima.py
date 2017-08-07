#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

start_value = tf.Variable(6.5)

parabola = start_value*start_value

minimizer = tf.train.AdamOptimizer(1.).minimize(parabola)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    [_, y, x]=sess.run([minimizer, parabola, start_value])
    if i%50==0:
        print("attempt {}, y={:.3e}, x={:.3e}".format(i,y,x))
