import tensorflow as tf
'''
w1=tf.constant([[-2.59392238,3.18602753,2.38825655],
 [-4.11017942,1.6826365,2.83427358]], name="w1")

w2=tf.constant([[-2.43003726],
 [ 3.33411145],
 [ 2.10067439]],name="w2")

x=tf.constant([[.1,.5],[1.,.0]],name="input")
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y))
'''

y_pred=[]

with tf.Session() as sess:
    saver=tf.train.Saver()
    saver.restore(sess,"model.ckpt")
    
    sess.run(y_pred,feed_dict={[[0.1,0.5],[0.5,0.6]]})
    print(y_pred)