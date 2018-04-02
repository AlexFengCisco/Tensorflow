import numpy as np
import tensorflow as tf
import matplotlib as mpl
from matplotlib import pyplot as plt
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat


learn = tf.contrib.learn


HIDDEN_SIZE = 30  
NUM_LAYERS = 2  
TIMESTEPS = 10  
BATCH_SIZE = 32 


TRAINING_STEPS = 3000  
TRAINING_EXAMPLES = 10000 
TESTING_EXAMPLES = 1000 
SAMPLE_GAP = 0.01  


def generate_data(seq):
    
    X = []
    y = []
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i:i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)



def LstmCell():
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
    return lstm_cell


def lstm_model(X, y):
    
    cell = tf.contrib.rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])

  
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = tf.reshape(output, [-1, HIDDEN_SIZE])


    predictions = tf.contrib.layers.fully_connected(output, 1, None)


    y = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])

   
    loss = tf.losses.mean_squared_error(predictions, y)

   
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.train.get_global_step(),
        optimizer='Adagrad',
        learning_rate=0.1)

    return predictions, loss, train_op



test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
train_X, train_y = generate_data(
    np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
test_X, test_y = generate_data(
    np.sin(
        np.linspace(test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))


regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir='model/'))


regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)


predicted = [[pred] for pred in regressor.predict(test_X)]


rmse = np.sqrt(((predicted - test_y)**2).mean(axis=0))
print('Mean Square Error is: %f' % (rmse[0]))


fig = plt.figure()
plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
plt.show()