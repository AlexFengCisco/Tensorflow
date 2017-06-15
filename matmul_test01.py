'''
test tensorflow matrix multiple , matrax a with m:n and matriax b with n:m , only two matrix has same n  value which is mathable ,or error will be reported 
'''
import tensorflow as tf
import numpy as np

a=tf.constant([[1,2],[3,4]],name="a")
b=tf.constant([[1,2,3],[4,5,6]],name="b")

c=tf.matmul(a, b)

init_op=tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)
  print(sess.run(c))

