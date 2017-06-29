import numpy as np
import matplotlib
matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([10],stddev=1,seed=1))

x1 =[1,2,3,4,5,6,7,8,9,10]

init =tf.global_variables_initializer()
t=np.arange(0.,5.,0.2)
with tf.Session() as sess:
    sess.run(init)
    w11=(sess.run(w1))



for i in range(100):
    plot_y=1

    plt.plot(i,plot_y,'bs')

plt.show()
