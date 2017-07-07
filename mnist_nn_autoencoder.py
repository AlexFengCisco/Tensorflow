import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from tensorflow.core.framework.cost_graph_pb2 import CostGraphDef
import tensorflow.examples.tutorials.mnist as mnist

LEARNING_RATE = 0.01
TRAINING_EPOCHS = 20
BATCH_SIZE = 256
DISPLAY_STEP = 1

EXAMPLES_TO_SHOW = 10

N_INPUT = 784
N_HIDDEN_1 = 256
N_HIDDEN_2 = 128

X = tf.placeholder("float",[None,N_INPUT])


weights ={
          'encoder_h1':tf.Variable(tf.random_normal([N_INPUT,N_HIDDEN_1])),
          'encoder_h2':tf.Variable(tf.random_normal([N_HIDDEN_1,N_HIDDEN_2])),
          'decoder_h1':tf.Variable(tf.random_normal([N_HIDDEN_2,N_HIDDEN_1])),
          'decoder_h2':tf.Variable(tf.random_normal([N_HIDDEN_1,N_INPUT]))
          }

biases = {
          'encoder_b1':tf.Variable(tf.random_normal([N_HIDDEN_1])),
          'encoder_b2':tf.Variable(tf.random_normal([N_HIDDEN_2])),
          'decoder_b1':tf.Variable(tf.random_normal([N_HIDDEN_1])),
          'decoder_b2':tf.Variable(tf.random_normal([N_INPUT]))
          }

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decoder_h2']),biases['decoder_b2']))
    return layer_2

encode_op = encoder(X)
decoder_op = decoder(encode_op)

y_pred = decoder_op

y_true = X 

cost = tf.reduce_mean(tf.pow(y_true - y_pred,2))
optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(cost)
    
with tf.Session() as sess:
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    init = tf.global_variables_initializer()
    sess.run(init)
    total_batch = int (mnist.train.num_examples/BATCH_SIZE)
    
    for epoch in range(TRAINING_EPOCHS):
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(BATCH_SIZE)
            _,c = sess.run([optimizer,cost],feed_dict={X:batch_xs})
            
        if epoch % DISPLAY_STEP == 0:
            print("Epoch:","%04d"%(epoch+1),'cost=',"{:.9f}".format(c))
    print("Optimization Finished!")
                
    encode_decode = sess.run(y_pred,feed_dict={X:mnist.test.images[:EXAMPLES_TO_SHOW]})
    
    f,a = plt.subplots(2,10,figsize=(10,2))
    for i in range(EXAMPLES_TO_SHOW):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[1][i].imshow(np.reshape(encode_decode[i],(28,28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
    
    
    
    
    