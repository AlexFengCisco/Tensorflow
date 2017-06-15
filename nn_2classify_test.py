import tensorflow as tf
from numpy.random import RandomState

BATCH_SIZE = 8
STEPS = 5000

w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x=tf.placeholder(tf.float32, shape=(None,2), name="x-input")
y_=tf.placeholder(tf.float32, shape=(None,1), name="y-input")
test_data=[[.1,.3],[.1,.3]]
test_Y=[0,1]
test_x=tf.placeholder(tf.float32, shape=(None,2), name="test_x-input")
#define neural network forward propagation
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

test_a=tf.matmul(test_x,w1)
y_pred=tf.matmul(test_a,w2)
#define loss and backward propagation 
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
cross_entropy_1 = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y_pred,1e-10,1.0)))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm=RandomState(1)
dataset_size=128
X=rdm.rand(dataset_size,2)

Y = [[int(x1+x2<1)] for (x1,x2) in X]


with tf.Session() as sess:
    
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    
    print(X)
    print(Y)
    print("====BEFORE TRAINING W1 W2")
    print(sess.run(w1))
    print(sess.run(w2))
    
    for i in range(STEPS+1):

        start=(i*BATCH_SIZE)%dataset_size
        end=min(start+BATCH_SIZE,dataset_size)
        
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training steps , cross entropy on all data is %g" %(i,total_cross_entropy))
    
    print("===AFTER TRAINING W1 W2")    
    print(sess.run(w1))
    print(sess.run(w2))
    print(y)
    
    saver=tf.train.Saver()
    saver.save(sess,"model.ckpt")
    print"====PREDICT"
    #print(sess.run(test_Y,feed_dict={test_x:test_data}))
    print("=====TEST CROSS ENTROPY")
    #total_cross_entropy_test = sess.run(cross_entropy_1,feed_dict={test_x:test_data,y_:Y})
    #print(total_cross_entropy_test)
    print(sess.run(w1))
    print(sess.run(w2))
            
            

 