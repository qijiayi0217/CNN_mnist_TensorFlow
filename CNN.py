import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 
 
BATCH_SIZE = 100
LR = 0.001

#preparing dataset
mnist = input_data.read_data_sets('./mnist', one_hot = True)
test_x = mnist.test.images[:]
test_y = mnist.test.labels[:]


## CNN structures
input_x = tf.placeholder(tf.float32, [None, 28*28])
image = tf.reshape(input_x, [-1, 28, 28, 1])
output_y = tf.placeholder(tf.int32, [None, 10])

conv1 = tf.layers.conv2d(
	inputs = image,
	filters = 16,
	kernel_size = 5,
	strides = 1,
	padding = 'same',
	activation = tf.nn.relu
	)

pool1 = tf.layers.max_pooling2d(
	conv1,
	pool_size = 2,
	strides = 2
	)

conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation = tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

#flaten the output from [n_sampes, 7, 7, 32] ---> [n_samples, 7*7*32]
flat = tf.reshape(pool2, [-1, 7*7*32])

#fully connect layer from [n_samples, 7*7*32] ---> [n_samples, 10]
# 10 means there are 10 kinds of digits
output = tf.layers.dense(flat, 10)

loss = tf.losses.softmax_cross_entropy(onehot_labels = output_y, logits = output)
train_step = tf.train.AdamOptimizer(LR).minimize(loss)

#This is for printing accuracy
accuracy = tf.metrics.accuracy(labels = tf.argmax(output_y, axis = 1), predictions = tf.argmax(output, axis = 1))[1]

sess = tf.Session()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

#training part
for epoch in range(20):
	for step in range(420):
		b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
		_, loss_ = sess.run([train_step, loss], {input_x: b_x, output_y: b_y})
		if step % 30 == 0:
			accuracy_ = sess.run(accuracy, {input_x: test_x, output_y: test_y})
			print("Epoch %d" % epoch, 'Step: ', step, "/420 | loss: %.4f" % loss_, " | test accuracy: %.2f" % accuracy_)


#try some predictions
test_output = sess.run(output, {input_x: test_x[:20]})
pred_y = np.argmax(test_output, 1)
print(pred_y, "prediction numbers")
print(np.argmax(test_y[:20], 1), 'real numbers')




