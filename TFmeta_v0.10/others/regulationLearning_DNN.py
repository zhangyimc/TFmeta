import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import loadDataset
import sys
from sklearn.utils import shuffle as SKshuffle

def normalize(array):
	return (array - array.mean()) / array.std()

with tf.device('/gpu:0'):
	outputDirectory = sys.argv[1]
	upstreamTrain = outputDirectory + 'upstreamGene.csv'
	downstreamTrain = outputDirectory + 'downstreamGene.csv'
	upstreamTest = outputDirectory + 'upstreamGeneTest.csv'
	downstreamTest = outputDirectory + 'downstreamGeneTest.csv'

	X = loadDataset.load_csv_with_header_single(filename=upstreamTrain, features_dtype=np.float)
	Y = loadDataset.load_csv_with_header_single(filename=downstreamTrain, features_dtype=np.float)

	train_size = int(0.9 * X.shape[0])
	np.random.seed(42)
	indices = np.random.permutation(X.shape[0])
	train_idx, test_idx = indices[:train_size], indices[train_size:]
	X_train, X_valid = [X.take(train_idx, axis=0), X.take(test_idx, axis=0)]
	Y_train, Y_valid = [Y.take(train_idx, axis=0), Y.take(test_idx, axis=0)]

	X_test = loadDataset.load_csv_with_header_single(filename=upstreamTest, features_dtype=np.float)
	Y_test = loadDataset.load_csv_with_header_single(filename=downstreamTest, features_dtype=np.float)

	for i in range(X_train.shape[0]):
		X_train[i] = normalize(X_train[i])
		Y_train[i] = normalize(Y_train[i])

	for i in range(X_valid.shape[0]):
		X_valid[i] = normalize(X_valid[i])
		Y_valid[i] = normalize(Y_valid[i])

	for i in range(X_test.shape[0]):
		X_test[i] = normalize(X_test[i])
		Y_test[i] = normalize(Y_test[i])

	X_train = np.concatenate((X_train, X_test))
	Y_train = np.concatenate((Y_train, Y_test))

	np.savetxt(outputDirectory + 'X_test.txt', X_test)
	np.savetxt(outputDirectory + 'Y_test.txt', Y_test)

	regu_coeff = 0.001
	learning_rate = 0.0005
	training_epochs = 20000
	batch_size = 128
	display_step = 10
	n_input = X_train.shape[1]
	n_output = Y_train.shape[1]

	def deepNN(x, dropout):
		with slim.arg_scope([slim.fully_connected], 
							activation_fn=tf.nn.tanh, 
							#normalizer_fn=slim.batch_norm, 
							weights_initializer=tf.truncated_normal_initializer(0.0, 0.01), 
							weights_regularizer=slim.l2_regularizer(0.0005)):
			net = slim.fully_connected(x, 128, scope='fc1')
			net = slim.dropout(net, dropout, scope='dropout1')
			for i in range(1):
				net = slim.fully_connected(net, 128, scope='fc%d' % (i+2))
				net = slim.dropout(net, dropout, scope='dropout%d' % (i+2))
			net = slim.fully_connected(net, n_output, activation_fn=None, normalizer_fn=None, scope='fc')
		return net

	x = tf.placeholder('float', [None, n_input])
	y = tf.placeholder('float', [None, n_output])
	dropout = tf.placeholder(tf.float32)

	pred = deepNN(x, dropout)
	loss = tf.losses.mean_squared_error(y, pred)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	MSE = tf.reduce_mean(tf.square(pred - y))

with tf.Session() as sess:
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())

	for epoch in range(training_epochs):
		X_train, Y_train = SKshuffle(X_train, Y_train, random_state=0)
		total_batch = int(X_train.shape[0] / batch_size)
		avg_loss = 0.
		for i in range(total_batch):
			batch_x = X_train[i*batch_size:(i+1)*batch_size]
			batch_y = Y_train[i*batch_size:(i+1)*batch_size]
			_, l, p = sess.run([optimizer, loss, pred], feed_dict={x: batch_x, y: batch_y, dropout: 0.9})
			avg_loss += l / total_batch

		if epoch % display_step == 0:
			print('Epoch:\t%5d\t,' % (epoch+1) + 
				'loss:\t%.12f\t,' % avg_loss + 
				'Validation_MSE:\t%.12f\t,' % MSE.eval(feed_dict={x: X_valid, y: Y_valid, dropout: 1.0}) + 
				'Testing_MSE:\t%.12f\t,' % MSE.eval(feed_dict={x: X_test, y: Y_test, dropout: 1.0}))

	print('Optimization Finished!')

	save_path = saver.save(sess, outputDirectory + 'model.ckpt')
	print('Model saved in file: %s' % save_path)

	# Validation
	print('Validation_MSE: %.12f ,' % MSE.eval(feed_dict={x: X_valid, y: Y_valid, dropout: 1.0}))

	# Testing
	print('Testing_MSE: %.12f ,' % MSE.eval(feed_dict={x: X_test, y: Y_test, dropout: 1.0}))
	Y_pred = pred.eval(feed_dict={x: X_test, dropout: 1.0})
	np.savetxt(outputDirectory + 'prediction.txt', Y_pred)
	print Y_pred
	print('\n')
	print Y_test

	tf.get_variable_scope().reuse_variables()
	np.savetxt(outputDirectory + 'l1-W.txt', tf.get_variable('fc1/weights').eval())
	np.savetxt(outputDirectory + 'l2-W.txt', tf.get_variable('fc2/weights').eval())
	np.savetxt(outputDirectory + 'lout-W.txt', tf.get_variable('fc/weights').eval())

	np.savetxt(outputDirectory + 'l1-B.txt', tf.get_variable('fc1/biases').eval())
	np.savetxt(outputDirectory + 'l2-B.txt', tf.get_variable('fc2/biases').eval())
	np.savetxt(outputDirectory + 'lout-B.txt', tf.get_variable('fc/biases').eval())
