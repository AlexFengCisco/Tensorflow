# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None
#FLAGS = "/tmp/tensorflow/mnist/input_data"

#define 5 layers neuro numbers ,the 1st layer is imput shapte 768
K = 200
L = 200
M = 60
N = 30


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  '''
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b
  '''
  pkeep=tf.constant(0.75,tf.float32)
  
  W1=tf.Variable(tf.truncated_normal([784,K],stddev=0.1))
  B1=tf.Variable(tf.zeros([K]))

  W2=tf.Variable(tf.truncated_normal([K,L],stddev=0.1))
  B2=tf.Variable(tf.zeros([L]))

  W3=tf.Variable(tf.truncated_normal([L,M],stddev=0.1))
  B3=tf.Variable(tf.zeros([M]))

  W4=tf.Variable(tf.truncated_normal([M,N],stddev=0.1))
  B4=tf.Variable(tf.zeros([N]))

  W5=tf.Variable(tf.truncated_normal([N,10],stddev=0.1))
  B5=tf.Variable(tf.zeros([10]))


  x=tf.placeholder(tf.float32,[None,784])

#changed from tf.nn.sigmoid to tf.nn.relu
  y1=tf.nn.relu(tf.matmul(x,W1)+B1)
  y11=tf.nn.dropout(y1,pkeep)
  y2=tf.nn.relu(tf.matmul(y11,W2)+B2)
  y22=tf.nn.dropout(y2,pkeep)
  y3=tf.nn.relu(tf.matmul(y22,W3)+B3)
  y33=tf.nn.dropout(y3,pkeep)
  y4=tf.nn.relu(tf.matmul(y33,W4)+B4)
  y44=tf.nn.dropout(y4,pkeep)
#  y=tf.nn.softmax(tf.matmul(y4,W5)+B5)
  y=tf.matmul(y44,W5)+B5
  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#  cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#  train_step = tf.train.GradientDescentOptimizer(0.03).minimize(cross_entropy)
  train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  timer=time.strftime("%A, %d %b %Y %H:%M:%S +0000")
  print(timer)
  train_begin_time=time.time()
  for _ in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
  train_end_time=time.time() 
  print("Training time ="+str(int(train_begin_time-train_end_time))+" seconds")
  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  test_end_time=time.time()
  print("Test time ="+str(int(train_end_time-test_end_time))+" seconds ")
  #print(sess.run(accuracy, feed_dict={x: mnist.test.images,
 #                                     y_: mnist.test.labels}))

  print(sess.run([accuracy,cross_entropy],feed_dict={x:mnist.test.images,
                                                     y_:mnist.test.labels}))
  sess.close()
if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  
