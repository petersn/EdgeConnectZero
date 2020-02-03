#!/usr/bin/python

import model
#model.set_dtype(model.tf.float32)
model.set_dtype(model.tf.float16)

import sys, signal, time, argparse
import edgeconnect_rules
import ctypes
import numpy as np
import engine
import link

parser = argparse.ArgumentParser(
	description="""
		Generates self-play games into the .json format using the C++ MCTS implementation.
		This program is much faster than generate_games.py, but can only do network self-play games using Tensorflow locally, and doesn't have all the extra features (like RPC evaluation, random play, UAI play, showing games as they progress, etc.).
		If you need one of those features, then use generate_games.py, and otherwise use this program.
	""",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--network", metavar="PATH", required=True, help="Path of the model to load.")
parser.add_argument("--output-games", metavar="PATH", required=True, help="Path to write .json games to. Writes in append mode, so it won't overwrite existing games.")
parser.add_argument("--visits", metavar="N", type=int, default=100, help="At each move in the self-play games perform MCTS until the root node has N visits.")
parser.add_argument("--buffer-size", metavar="N", type=int, default=64, help="Use a buffer that evaluates N samples in parallel on the GPU. Requires that we launch 2*N threads each playing a parallel game, and thus have memory usage proportional to N.")
args = parser.parse_args()
print("Arguments:", args)

engine.initialize_model(args.network)

work_buffers = [
	np.zeros((args.buffer_size, edgeconnect_rules.BOARD_SIZE, edgeconnect_rules.BOARD_SIZE, model.Network.INPUT_FEATURE_COUNT), dtype=np.float32)
	for _ in (0, 1)
]
link.launch_threads(
	args.output_games.encode("utf-8"),
	args.visits,
	ctypes.c_void_p(work_buffers[0].ctypes.data),
	ctypes.c_void_p(work_buffers[1].ctypes.data),
	args.buffer_size,
	args.buffer_size * 2,
)

def handler(signal, frame):
	print("Exiting cleanly...", end=' ')
	sys.stdout.flush()
	link.shutdown()
	print("all threads shutdown.")
	exit()
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

workload_times = [time.time()]
averaging_window = 1000

# We have to keep references to recent arrays around so the C++ side doesn't accidentally try to read from them after they become invalid.
recent_arrays = []

class EWMA:
	def __init__(self):
		self.value = 0
		self.alpha = 0.995

	def update(self, x):
		self.value = self.alpha * self.value + (1 - self.alpha) * x

get_workload_delays = EWMA()
sess_run_delays = EWMA()
complete_workload_delays = EWMA()
convert_to_delays = EWMA()
convert_from_delays = EWMA()

#for work_buffer in work_buffers:
#	print("Work buffer:", work_buffer.dtype, work_buffer.shape, work_buffer.strides, work_buffer.flags.c_contiguous)

#block_input = np.random.randn(128, 23, 23, engine.network.INPUT_FEATURE_COUNT)

while True:
	start = time.time()
	workload_index = link.get_workload()
	get_workload_delays.update(time.time() - start)

	start = time.time()
	features = work_buffers[workload_index]
#	features = work_buffers[workload_index].astype(np.float16)
#	assert features.flags.c_contiguous
	convert_to_delays.update(time.time() - start)
#	features = block_input

	start = time.time()
	posteriors, values = engine.sess.run(
		[engine.network.policy_output, engine.network.value_output],
		feed_dict={
			engine.network.input_ph: features,
			engine.network.is_training_ph: False,
		},
	)
#	if "posteriors" not in globals():
#		posteriors = np.random.randn(128, 23, 23, 1).astype(np.float32)
#		values = np.random.random((128, 1)).astype(np.float32)
	sess_run_delays.update(time.time() - start)
	if False:
		start = time.time()
		posteriors = posteriors.astype(np.float32)
		values = values.astype(np.float32)
		convert_from_delays.update(time.time() - start)
	assert posteriors.dtype == np.float32
	assert posteriors.flags.c_contiguous
	assert values.dtype == np.float32
	assert values.flags.c_contiguous
	recent_arrays.append((posteriors, values))
	# XXX: TODO: Worry a lot about whether or not `work_buffers`, `posteriors` and `values` are packed contiguously with the right stride ordering!
	# I don't currently check this, and this is really important.
	start = time.time()
	link.complete_workload(workload_index, ctypes.c_void_p(posteriors.ctypes.data), ctypes.c_void_p(values.ctypes.data))
	complete_workload_delays.update(time.time() - start)

	workload_times.append(time.time())
	span = workload_times[-averaging_window:]
	rate = (len(span) - 1.0) / (span[-1] - span[0])
	if len(workload_times) % 2000 == 0:
		print("Rate: %.3fk evals/s  (Total: %ik) get_workload: %.3fms - sess.run: %.3fms - complete_workload: %.3fms - fp32->fp16: %.3fms fp16->fp32: %.3fms" % (
			rate * args.buffer_size * 1e-3,
			len(workload_times) * args.buffer_size * 1e-3,
			1e3 * get_workload_delays.value,
			1e3 * sess_run_delays.value,
			1e3 * complete_workload_delays.value,
			1e3 * convert_to_delays.value,
			1e3 * convert_from_delays.value,
		))

	# Technically keeping just the two most recent should be sufficient to avoid any issues.
	recent_arrays = recent_arrays[-6:]

