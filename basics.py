#!/usr/bin/env python3
import tensorflow as tf

x=tf.Variable(0.5)
y = x*x
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("x =",sess.run(x))
print("y =",sess.run(y))
