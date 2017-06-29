import tensorflow as tf
import os
import matplotlib.image as mp_image
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("Dataset/", one_hot=True)

features1 = 32
features2  = 64
neuron_1 = 100
num_step = 100
batch_size = 100
num_batch = mnist.train.num_examples // batch_size + 1

x = tf.placeholder(shape=[None, 32*64], dtype=tf.float32)
y_correct = tf.placeholder(shape=[None, 10], dtype=tf.float32)

x_image = tf.reshape(x, shape=[-1, 32, 64, 1])

W_conv1 = tf.Variable(tf.truncated_normal(shape=[4, 4, 1, features1], stddev=0.1, dtype=tf.float32))
b_conv1 = tf.Variable(tf.truncated_normal(shape=[features1], stddev=0.1, dtype=tf.float32))
z_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
a_conv1 = tf.nn.max_pool(z_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv2 = tf.Variable(tf.truncated_normal(shape=[4, 4, features1, features2], stddev=0.1, dtype=tf.float32))
b_conv2 = tf.Variable(tf.truncated_normal(shape=[features2], stddev=0.1, dtype=tf.float32))
z_conv2 = tf.nn.relu(tf.nn.conv2d(a_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
a_conv2 = tf.nn.max_pool(z_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

a_conv2_flat = tf.reshape(a_conv2, shape=[-1, 8*16*features2])



W_1 = tf.Variable(tf.truncated_normal([8*16*features2, neuron_1],stddev=0.1, dtype=tf.float32))
b_1 = tf.Variable(tf.truncated_normal([neuron_1], stddev=0.1, dtype=tf.float32))
z_1 = tf.matmul(a_conv2_flat, W_1) + b_1
a_1 = tf.nn.relu(z_1)

W = tf.Variable(tf.truncated_normal([neuron_1, 10],stddev=0.1, dtype=tf.float32))
b = tf.Variable(tf.truncated_normal([10], stddev=0.1, dtype=tf.float32))
y = tf.matmul(a_1, W) + b


cost = tf.nn.softmax_cross_entropy_with_logits(labels=y_correct, logits=y)



train_step = tf.train.AdamOptimizer(0.0001).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_correct, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
try:
    print('Loading model from file...')
    saver.restore(sess, "D:\Final_project\Checkpoints\model.bin")
    #sess.run(tf.global_variables_initializer())
    #for i in range(num_step):
    #    for j in range(20):
    #        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #       sess.run(train_step, feed_dict={x: batch_xs, y_correct: batch_ys})
    #    valid_score = sess.run(accuracy,
    #                           feed_dict={x: mnist.validation.images[0:200], y_correct: mnist.validation.labels[0:200]})
    #    print("Step {}: {}".format(i, valid_score))
    #test_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_correct: mnist.test.labels})
    #print("Test score: {}".format(test_score))
    #saver.save(sess, "D:\Final_project\Checkpoints\model4.bin")
except:
    print('The model file does not exist!')
    sess.run(tf.global_variables_initializer())
    for i in range(num_step):
        for j in range(20):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_correct: batch_ys})
        valid_score = sess.run(accuracy,feed_dict={x: mnist.validation.images[0:200], y_correct: mnist.validation.labels[0:200]})
        print("Step {}: {}".format(i, valid_score))
    test_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_correct: mnist.test.labels})
    print("Test score: {}".format(test_score))
    saver.save(sess, "D:\Final_project\Checkpoints\model.bin")

print('Try to run the accuracy function after loading from file')
test_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_correct: mnist.test.labels})
print("Test score: {}".format(test_score))

files = os.listdir("Images")
for file in files:
    tmp = mp_image.imread("Images/" + file)
    input_image = tmp
    plt.imshow(input_image, cmap='gray')
    plt.show()
    input_image = input_image.reshape([1, 32*64])
    tmp2 = sess.run(y, feed_dict={x: input_image})
    tmp3 = sess.run(tf.argmax(tmp2, axis=1))
    print(tmp3)


