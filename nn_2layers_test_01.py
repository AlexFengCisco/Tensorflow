'''
test tensorflow matrix multiple , matrax a with m:n and matriax b with n:m , only two matrix has same n value which is mathable ,or error will be reported 
'''

import tensorflow as tf
import numpy as np

w1=tf.Variable(tf.random_uniform([2,3],minval=-1,maxval=1,name="w1"))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,name="w2"))

x=tf.placeholder(tf.float32, shape=(3,2), name="input")
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

sess=tf.Session()
init_op=tf.global_variables_initializer()
sess.run(init_op)

print(sess.run(w1))
print(sess.run(w2))
print(sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))

sess.close()