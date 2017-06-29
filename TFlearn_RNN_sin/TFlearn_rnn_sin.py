import numpy as np
import tensorflow as tf

import matplotlib as mpl
from lib2to3.fixer_util import Number
from numpy import dtype
mpl.use("TKagg")
from matplotlib import pyplot as plt

learn = tf.contrib.learn

HIDDEN_SIZE = 30
NUM_LAYERS = 2 # LSTM layers Number
TIMETEPS = 10 #RNN fragment length 
TRAINING_STEPS = 10000
BATCH_SIZE = 1000

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01

def generate_data(seq):
    X = []
    y = []
    for i in range(len(seq)-TIMETEPS-1):
        X.append([seq[i:i+TIMETEPS]])
        y.append([seq[i+TIMETEPS]])
    return np.array(X,dtype=np.float32),np.array(y,dtype=np.float32)

def lstm_model(X,y):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    cell = tf.nn.rnn_cell.MulRNNCell([lstm_cell]*NUM_LAYERS)
    X_ = tf.unpack(X,axis=1)
    output,_ = tf.nn.rnn(cell,x_,dtype=float32)
    output = output[-1]
    prediction,loss = learn.models.linear_regression(output,y)
    
    train_op = tf.contrib.layers.optimize_loss(loss,tf.contrib.framework.get_global_step(),optimizer="Adagrad",learning_rate=0.1)
    
    return prediction,loss,train_op

regressor = learn.Estimator(model_fn=lstm_model)

test_start = TRAINING_EXAMPLES*SAMPLE_GAP
test_end = (TRAINING_EXAMPLES+TESTING_EXAMPLES)*SAMPLE_GAP

train_X,train_y = generate_data(np.sin(np.linspace(0,test_start,TRAINING_EXAMPLES,dtype=np.float32)))
test_X,test_y = generate_data(np.sin(np.linspace(test_start,test_end,TESTING_EXAMPLES,dtype=np.float32)))

regressor.fit(train_X,train_y,batch_size=BATCH_SIZE,steps=TRAINING_STEPS)

predicted = [[pred] for pred in regressor.predict(test_X)]

rmse = np.sqrt((predicted-test_y)**2).mean(axis=0)
print("Mean Square Error is : %f"%rmse[0])

fig=plt.figure()
plot_predicted = plt.plot(predicted,label='predicted')
plot_test = plt.plot(test_y,label='real_sin')
plt.legend([plot_predicted,plot_test],['predicted','real_sin'])
plt.show()