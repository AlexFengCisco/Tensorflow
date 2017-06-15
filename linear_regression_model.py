import tensorflow as tf
import numpy as np
import pylab

x_data=np.random.rand(100).astype(np.float32)
noise=np.random.normal(scale=0.01,size=len(x_data))
y_data=x_data*0.1+0.3+noise

pylab.plot(x_data,y_data,'.')
print("=================target y_data")
print y_data

W=tf.Variable(tf.random_uniform([1],0.0,1.0),name="Alex_Weight")
b=tf.Variable(tf.zeros([1]),name="Alex_bias")
y=W*x_data+b

print("================random original W (weight) and b (bias)")
print (W)
print (b)

loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)
#init=tf.initialize_all_variables()
init=tf.global_variables_initializer()

print("loss",loss)
print("optimizer",optimizer)
print("train",train)
print(init)

#print(tf.get_default_graph().as_graph_def())

sess=tf.Session()
sess.run(init)
y_initial_values=sess.run(y)

print y
print ("======y_initial_values")
print y_initial_values
print(sess.run([W,b]))

for step in range(22201):
    sess.run(train)


print(sess.run([W,b]))
print(sess.run(W))
print(sess.run(b))

sess.close()

'''
pylab.plot(x_data,y_data,'.',label="target_values")
pylab.plot(x_data,y_initial_values,'.',label="initial_values")
pylab.plot(x_data,sess.run(y),'.',label="trained_values")
pylab.legend()
pylab.ylim(0,1.0)
'''
