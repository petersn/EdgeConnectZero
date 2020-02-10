#!/usr/bin/python

import sys, time
import numpy as np
import tensorflow as tf
from functools import reduce
from edgeconnect_rules import BOARD_SIZE, BOARD_RADIUS

product = lambda l: reduce(lambda x, y: x * y, l, 1)

MOVE_TYPES = 1

def set_dtype(dtype):
	global DTYPE, NP_DTYPE
	DTYPE = dtype
	NP_DTYPE = {
		tf.float16: np.float16,
		tf.float32: np.float32,
	}[DTYPE]

set_dtype(tf.float32)

def gelu(x):
	return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

class Network:
	INPUT_FEATURE_COUNT = 12
	NONLINEARITY = [tf.nn.relu]
	FILTERS = 64 #128
	CONV_SIZE = 3
	BLOCK_COUNT = 12
	VALUE_FILTERS = 1
	VALUE_FC_SIZE = 32
	USE_LEAKY_FC = False # True
#	VALUE_FC_SIZES = [BOARD_SIZE * BOARD_SIZE * VALUE_FILTERS, 32, 1]
	POLICY_OUTPUT_SHAPE = [None, BOARD_SIZE, BOARD_SIZE, MOVE_TYPES]
	VALUE_OUTPUT_SHAPE = [None, 1]

	total_parameters = 0

	def __init__(self, scope_name, build_training=False):
		self.scope_name = scope_name
		with tf.variable_scope(scope_name):
			# First we build a tower producing some number of features.
			self.build_tower()
			# Next we build the heads.
			self.build_policy_head()
			self.build_value_head()
		# Finally, we build the training components if requested.
		if build_training:
			self.build_training()

	def build_tower(self):
		# Construct input/output placeholders.
		self.input_ph = tf.placeholder(
			tf.float32,
			shape=[None, BOARD_SIZE, BOARD_SIZE, self.INPUT_FEATURE_COUNT],
			name="input_placeholder")
		self.desired_policy_ph = tf.placeholder(
			DTYPE,
			shape=self.POLICY_OUTPUT_SHAPE,
			name="desired_policy_placeholder")
		self.desired_value_ph = tf.placeholder(
			DTYPE,
			shape=self.VALUE_OUTPUT_SHAPE,
			name="desired_value_placeholder")
		self.learning_rate_ph = tf.placeholder(DTYPE, shape=[], name="learning_rate")
		self.is_training_ph = tf.placeholder(tf.bool, shape=[], name="is_training")

		# Begin constructing the data flow.
		self.parameters = []
		self.flow = self.input_ph
		if DTYPE == tf.float16:
			self.flow = tf.cast(self.flow, tf.float16)
		# Stack an initial convolution.
		self.stack_convolution(self.CONV_SIZE, self.INPUT_FEATURE_COUNT, self.FILTERS)
		self.stack_nonlinearity()
		# Stack some number of residual blocks.
		for _ in range(self.BLOCK_COUNT):
			self.stack_block()
		# Stack a final 1x1 convolution transitioning to fully-connected features.
		#self.stack_convolution(1, self.FILTERS, self.OUTPUT_CONV_FILTERS, batch_normalization=False)

	def build_policy_head(self):
		weights = self.new_weight_variable([1, 1, self.FILTERS, MOVE_TYPES])
		self.policy_output = tf.nn.conv2d(self.flow, weights, strides=[1, 1, 1, 1], padding="SAME", name="policy_output")
#		self.policy_output = tf.reshape(self.policy_output, [-1, BOARD_SIZE * BOARD_SIZE, MOVE_TYPES])
#		self.policy_output = tf.matrix_transpose(self.policy_output)
		if DTYPE == tf.float16:
			self.policy_output = tf.cast(self.policy_output, tf.float32)
		self._po = tf.identity(self.policy_output, name="policy_output_id")

	def build_value_head(self):
		weights = self.new_weight_variable([1, 1, self.FILTERS, self.VALUE_FILTERS])
		value_layer = tf.nn.conv2d(self.flow, weights, strides=[1, 1, 1, 1], padding="SAME")
		value_layer = tf.reshape(value_layer, [-1, BOARD_SIZE * BOARD_SIZE * self.VALUE_FILTERS])

		fc_w1 = self.new_weight_variable([BOARD_SIZE * BOARD_SIZE, self.VALUE_FC_SIZE])
		fc_b1 = self.new_bias_variable([self.VALUE_FC_SIZE])
		if self.USE_LEAKY_FC:
			value_hidden = tf.nn.leaky_relu(
				features=tf.matmul(value_layer, fc_w1) + fc_b1,
				alpha=0.01,
			)
		else:
			value_hidden = self.NONLINEARITY[0](tf.matmul(value_layer, fc_w1) + fc_b1)

		fc_w2 = self.new_weight_variable([self.VALUE_FC_SIZE, 1])
		fc_b2 = self.new_bias_variable([1])
		self.value_output = tf.nn.tanh(tf.matmul(value_hidden, fc_w2) + fc_b2, name="value_output")
		if DTYPE == tf.float16:
			self.value_output = tf.cast(self.value_output, tf.float32)
		self._vo = tf.identity(self.value_output, name="value_output_id")

	def build_training(self):
		# Make policy head loss.
#		assert self.desired_policy_ph.shape == self.policy_output.shape
		self.policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
			labels=tf.reshape(self.desired_policy_ph, [-1, BOARD_SIZE * BOARD_SIZE * MOVE_TYPES]),
			logits=tf.reshape(self.policy_output, [-1, BOARD_SIZE * BOARD_SIZE * MOVE_TYPES]),
		))
		# Make value head loss.
		self.value_loss = tf.reduce_mean(tf.square(self.desired_value_ph - self.value_output))
		# Make regularization loss.
		regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
		reg_variables = tf.trainable_variables(scope=self.scope_name)
		self.regularization_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
		# Loss is the sum of these three.
		self.loss = 1.0 * self.policy_loss + self.value_loss + self.regularization_term

		# Associate batch normalization with training.
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.scope_name)
		with tf.control_dependencies(update_ops):
#			self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).minimize(self.loss)
			self.train_step = tf.train.MomentumOptimizer(
				learning_rate=self.learning_rate_ph, momentum=0.9).minimize(self.loss)

	def new_weight_variable(self, shape):
		self.total_parameters += product(shape)
		stddev = 0.2 * (2.0 / product(shape[:-1]))**0.5
		var = tf.Variable(tf.truncated_normal(shape, stddev=stddev, dtype=DTYPE), dtype=DTYPE)
		self.parameters.append(var)
		return var

	def new_bias_variable(self, shape):
		self.total_parameters += product(shape)
		var = tf.Variable(tf.constant(0.001, shape=shape, dtype=DTYPE), dtype=DTYPE)
		self.parameters.append(var)
		return var

	def stack_convolution(self, kernel_size, old_size, new_size, batch_normalization=True):
		weights = self.new_weight_variable([kernel_size, kernel_size, old_size, new_size])
		self.flow = tf.nn.conv2d(self.flow, weights, strides=[1, 1, 1, 1], padding="SAME")
		if batch_normalization:
			self.flow = tf.layers.batch_normalization(
				self.flow,
				center=True,
				scale=True,
				training=self.is_training_ph)
		else:
			bias = self.new_bias_variable([new_size])
			self.flow = self.flow + bias # TODO: Is += equivalent?

	def stack_nonlinearity(self):
		self.flow = self.NONLINEARITY[0](self.flow)
#		self.flow = tf.nn.leaky_relu(
#			features=self.flow,
#			alpha=0.01,
#		)

	def stack_block(self):
		initial_value = self.flow
		# Stack the first convolution.
		self.stack_convolution(self.CONV_SIZE, self.FILTERS, self.FILTERS)
		self.stack_nonlinearity()
		# Stack the second convolution.
		self.stack_convolution(self.CONV_SIZE, self.FILTERS, self.FILTERS)
		# Add the skip connection.
		self.flow = self.flow + initial_value
		# Stack on the deferred non-linearity.
		self.stack_nonlinearity()

	def train(self, samples, learning_rate):
		self.run_on_samples(self.train_step, samples, learning_rate=learning_rate, is_training=True)

	def get_loss(self, samples):
		return self.run_on_samples(self.loss, samples)

	def get_accuracy(self, samples):
		results = self.run_on_samples(self.final_output, samples).reshape((-1, 64 * 64))
		#results = results.reshape((-1, 64 * 8 * 8))
		results = np.argmax(results, axis=-1)
		assert results.shape == (len(samples["features"]),)
		correct = 0
		for move, result in zip(samples["moves"], results):
			lhs = np.argmax(move.reshape((64 * 64,)))
			#assert lhs.shape == result.shape == (2,)
			correct += lhs == result #np.all(lhs == result)
		return correct / float(len(samples["features"]))

	def run_on_samples(self, f, samples, learning_rate=0.01, is_training=False):
		return sess.run(f, feed_dict={
			self.input_ph:          samples["features"],
			self.desired_policy_ph: samples["policies"],
			self.desired_value_ph:  samples["values"],
			self.learning_rate_ph:  learning_rate,
			self.is_training_ph:    is_training,
		})

# XXX: This is horrifically ugly.
# TODO: Once I have a second change it to not do this horrible graph scraping that breaks if you have other things going on.
def get_batch_norm_vars(net):
	return [
		i for i in tf.global_variables(scope=net.scope_name)
		if "batch_normalization" in i.name and ("moving_mean:0" in i.name or "moving_variance:0" in i.name)
	]

def save_model(net, path):
	x_conv_weights = [sess.run(var) for var in net.parameters]
	x_bn_params = [sess.run(i) for i in get_batch_norm_vars(net)]
	np.save(path, [x_conv_weights, x_bn_params])
	print("\x1b[35mSaved model to:\x1b[0m", path)

# XXX: Still horrifically fragile wrt batch norm variables due to the above horrible graph scraping stuff.
def load_model(net, path):
	x_conv_weights, x_bn_params = np.load(path, allow_pickle=True)
	assert len(net.parameters) == len(x_conv_weights), "Parameter count mismatch!"
	operations = []
	for var, value in zip(net.parameters, x_conv_weights):
		operations.append(var.assign(value))
	bn_vars = get_batch_norm_vars(net)
	assert len(bn_vars) == len(x_bn_params), "Bad batch normalization parameter count!"
	for var, value in zip(bn_vars, x_bn_params):
		operations.append(var.assign(value))
	sess.run(operations)

class EWMA:
	def __init__(self):
		self.value = 0
		self.alpha = 0.995

	def update(self, x):
		self.value = self.alpha * self.value + (1 - self.alpha) * x

if __name__ == "__main__":
	set_dtype(tf.float16)
	net = Network("net/")
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	load_model(net, "run-cm5-r=11-f=64-b=12-fc=32-g=500-v=800-distl-t0/models/model-343.npy")
	print("About to save!")
	saver = tf.train.Saver()
	saver.save(sess, "cpp/checkpoints/edgeconnect-model")
	exit()

if __name__ == "__main__":
	set_dtype(tf.float16)
	net = Network("net/")
#	print(get_batch_norm_vars(net))
	print("Parameter count:", net.total_parameters)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	load_model(net, "run-cm5-r=11-f=64-b=12-fc=32-g=500-v=800-distl-t0/models/model-260.npy")
	batch_size = 1
	block_input = np.random.randn(batch_size, BOARD_SIZE, BOARD_SIZE, Network.INPUT_FEATURE_COUNT).astype(NP_DTYPE)
	print("block_input:", block_input.dtype, block_input.shape, block_input.strides, block_input.flags.c_contiguous)
	print("Warming up.")
	for _ in range(10):
		sess.run((net.policy_output, net.value_output), feed_dict={net.input_ph: block_input, net.is_training_ph: False})
	print("Main run.")
	COUNT = 200
	delay = EWMA()
	start = time.time()
	for _ in range(COUNT):
		rs = time.time()
		sess.run((net.policy_output, net.value_output), feed_dict={net.input_ph: block_input, net.is_training_ph: False})
		delay.update(time.time() - rs)
	end = time.time()
	print("Took:", end - start)
	rate = batch_size * COUNT / (end - start)
	print("Evaluation rate:", rate)

	summary_string = f"dtype={DTYPE} batch_size={batch_size:3} board_radius={BOARD_RADIUS} filters={net.FILTERS} blocks={net.BLOCK_COUNT} evals/s={rate} delay={1e3 * delay.value:.2}ms"
	print(summary_string)
	with open("bench", "a+") as f:
		print(summary_string, file=f)

