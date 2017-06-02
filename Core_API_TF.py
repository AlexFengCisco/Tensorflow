'''
Created on May 7, 2017

@author: AlexFeng
'''

import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
tf.summary.scalar("loss",loss)
tf.summary.histogram("W",W)
tf.summary.histogram("b",b)
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
merged = tf.summary.merge_all()
test_writer = tf.summary.FileWriter("log/test1")
init = tf.global_variables_initializer()
sess = tf.Session()
test_writer.add_graph(sess.graph)
sess.run(init) # reset values to wrong
for i in range(1000):
  if i%5 == 0:
    s=sess.run(merged,feed_dict={x:x_train,y:y_train})
    test_writer.add_summary(s,i)
  sess.run(train, {x:x_train, y:y_train})
  #summary=sess.run(merged)
  #test_writer.add_summary(loss)

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

#writer = tf.summary.FileWriter("log/test",sess.graph)
#writer.close()

