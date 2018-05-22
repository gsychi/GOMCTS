#THIS IS A NEURAL NETWORK WITH MNIST DATA
#will wire in stuff for specific tic tac toe/go once available

# will work on making this convolutional...
# this is beneficial for computer AIs because it starts to understand certain patterns in the board.
# may not be necessary for tic tac toe but oh well

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


#input + output size for nn
inputSize = 784
nClasses = 10

nodes1 = 800
nodes2 = 500
nodes3 = 300
nodes4 = 100

batchSize = 100 #smaller = more accurate? :\

x = tf.placeholder('float', [None, inputSize])
y = tf.placeholder('float')

def nn_model(data):
    h1Layer = {'weights': tf.Variable(tf.random_normal([inputSize, nodes1])),
               'biases': tf.Variable(tf.random_normal([nodes1]))}

    h2Layer = {'weights': tf.Variable(tf.random_normal([nodes1, nodes2])),
               'biases': tf.Variable(tf.random_normal([nodes2]))}

    h3Layer = {'weights': tf.Variable(tf.random_normal([nodes2, nodes3])),
               'biases': tf.Variable(tf.random_normal([nodes3]))}

    h4Layer = {'weights': tf.Variable(tf.random_normal([nodes3, nodes4])),
               'biases': tf.Variable(tf.random_normal([nodes4]))}

    outputLayer = {'weights': tf.Variable(tf.random_normal([nodes4, nClasses])),
                   'biases': tf.Variable(tf.random_normal([nClasses]))}

    l1 = tf.add(tf.matmul(data, h1Layer['weights']), h1Layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, h2Layer['weights']), h2Layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, h3Layer['weights']), h3Layer['biases'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, h4Layer['weights']), h4Layer['biases'])
    l4 = tf.nn.relu(l4)

    output = tf.matmul(l4, outputLayer['weights']) + outputLayer['biases']

    return output

def train_nn(c, epochs):
    prediction = nn_model(c)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = epochs

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # newTraining is a numpy array
        newTraining = np.ones((1, 784))
        newTraining[0] = mnist.train.images[0]

        #initial predictions
        print("BASED ON MODEL:")
        print(sess.run(prediction, feed_dict={x: newTraining}))

        #training
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples/batchSize)):
                epochX, epochY = mnist.train.next_batch(batchSize) #this is only for the data here, for self data, this has to be changed
                i, d = sess.run([optimizer, cost], feed_dict={x: epochX, y: epochY})
                epoch_loss += d
            print('Epoch', epoch, 'completed out of', hm_epochs, ' || loss:', epoch_loss)

            # checking information
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

            print("BASED ON MODEL:")
            print(sess.run(prediction, feed_dict={x: newTraining}))


train_nn(x, 50)


# THIS PRINTS DIRECTLY FROM A TENSOR

newTraining = np.ones((1, 784))
newTraining[0] = mnist.train.images[0]
data_new = np.asarray(newTraining, np.float32)
data_tf = tf.convert_to_tensor(data_new, np.float32)
blah = nn_model(data_tf)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    #print(sess.run(blah))





