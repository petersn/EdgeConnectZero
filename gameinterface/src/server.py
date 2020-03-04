#!/usr/bin/python

AI_PLAYER_INDEX = 2

import logging
logging.basicConfig(
	filename="/home/snp/tmp/GAME_LOG",
	format="[%(asctime)s] %(message)s",
	level=logging.INFO,
)
logging.info("Initializing server. AI_PLAYER_INDEX=%r", AI_PLAYER_INDEX)

import sys
sys.path.append("../..")
import engine
import edgeconnect_rules
import uai_ringmaster

import json, copy, random, threading, queue, time
import tornado.ioloop
import tornado.web
import tornado.websocket
from tornado import gen

import subprocess

class EngineThread(threading.Thread):
	def __init__(self):
		super().__init__()

#		engine.setup_evaluator()

#		engine.initialize_model("/home/snp/proj/EdgeConnectZero/run1/models/model-037.npy")
#		engine.initialize_model("/home/snp/proj/EdgeConnectZero/run5-r=5-f=32-b=8/models/model-019.npy")
#		engine.initialize_model("/home/snp/proj/EdgeConnectZero/run6-r=3-f=32-b=4/models/model-030.npy")
#		engine.initialize_model("/home/snp/proj/EdgeConnectZero/run7-r=3-f=32-b=4-g=1000-v=1000-fixup-cpp/models/model-009.npy")

#		engine.initialize_model("/tmp/model-170.npy")
#		self.mcts = engine.MCTSEngine()

		self.player = uai_ringmaster.UAIPlayer(cmd=["python3", "../../uai_interface.py", "--network-path", "/home/snp/tmp/model-343.npy"])

#		self.player = uai_ringmaster.UAIPlayer(
#			cmd=[
#				"ssh",
#					"-p", "44767",
#					"66.31.30.53",
#					"cd EdgeConnectZero; python uai_interface.py --fast",
#			],
#		)
		print(self.player.showboard())

		self.message_queue = queue.Queue()
		self.evaluation_curve = {}

	def run(self):
		while True:
			web_handler, state = self.message_queue.get()
			print("Request:", web_handler)
			print(state)
			logging.info("State just before AI move: %s", state.to_string())
			#self.mcts.set_state(state)
			self.player.set_state(state)
			move_count = 2 if state.move_state[1] == "a" else 1
			for _ in range(move_count):
				#move = self.mcts.genmove(0.1, use_weighted_exponent=2.0)
				move = self.player.genmove(1000)
#				move = self.player.genmove(5000)
#				move = random.choice(state.legal_moves())
				logging.info("AI move: %r", move)
				if move == "pass":
					break
				state.make_move(move)
				#self.mcts.set_state(state)
				self.player.set_state(state)
			logging.info("Final state after AI move: %s", state.to_string())
			web_handler.update_queue.put({
				"playerToMove": int(state.move_state[0]),
				"playerMoveIndex": state.move_state[1],
				"board": {qr: int(state.board[qr]) for qr in edgeconnect_rules.ALL_VALID_QR},
				"nps": self.player.nps,
				"evaluation": self.player.evaluation,
			})
			#web_handler.player_to_move = 1
			#web_handler.player_move_index = "a"
			#web_handler.board = 
			#web_handler.send_update()

class WebSocketHandler(tornado.websocket.WebSocketHandler):
	def check_origin(self, origin):
		return True

	def open(self):
		logging.info("NEW CLIENT")
		print("WebSocket opened to:", self)
		self.player_to_move = 1
		self.player_move_index = "b"
		self.board = {qr: 0 for qr in edgeconnect_rules.ALL_VALID_QR}
		self.old_board = self.board.copy()
		self.game_over = False
		self.is_thinking = 0
		self.final_state_to_return = None
		self.nps = -1
		self.evaluation_curve = {}
		self.update_queue = queue.Queue()
		self.send_update()

	def make_state(self):
		if self.final_state_to_return is not None:
			return self.final_state_to_return
		state = edgeconnect_rules.EdgeConnectState.initial()
		state.move_state = self.player_to_move, self.player_move_index
		for qr, value in self.board.items():
			state.board[qr] = value
		return state

	def on_message(self, message):
		payload = json.loads(message)
		should_check_for_ai_move = False
		force_move = False
		if payload["kind"] == "click" and not self.game_over:
			qr = tuple(payload["qr"])
			if self.board[qr] != 0:
				return
			# Only let the human play one side.
			if self.player_to_move != 3 - AI_PLAYER_INDEX:
				return
			logging.info("Player move: %r", qr)
			self.board[qr] = self.player_to_move
			if self.player_move_index == "a":
				self.player_move_index = "b"
			else:
				self.player_to_move = 3 - self.player_to_move
				self.player_move_index = "a"
			should_check_for_ai_move = True
			self.send_update()
		if payload["kind"] == "ping":
			if self.game_over:
				self.write_message(json.dumps({"kind": "ping"}))
			else:
				self.send_update()
		if payload["kind"] == "resume":
			state = edgeconnect_rules.EdgeConnectState.from_string(payload["boardString"].strip())
			print("Resumed from:", payload["boardString"])
			self.board = {qr: int(state.board[qr]) for qr in self.board}
			self.old_board = self.board.copy()
			self.player_to_move, self.player_move_index = state.move_state
			logging.info("Resume: %s", state.to_string())
			self.send_update(from_resume=True)
		if payload["kind"] == "genmove":
			force_move = True
			should_check_for_ai_move = True
		if should_check_for_ai_move:
			if force_move or (self.player_to_move == AI_PLAYER_INDEX and self.player_move_index == "a"):
				state = self.make_state()
				if state.result_with_early_stopping() is None:
					engine_thread.message_queue.put((self, state))
					self.is_thinking += 1
					self.send_update()

		if self.update_queue.qsize() and not self.game_over:
			new_values = self.update_queue.get()
#			print("Update queue:", new_values)
			self.player_to_move = new_values["playerToMove"]
			self.player_move_index = new_values["playerMoveIndex"]
			self.old_board = self.board.copy()
			self.is_thinking -= 1
			self.board = new_values["board"]
			self.nps = new_values["nps"]
			self.evaluation_curve[sum(v != 0 for v in self.board.values())] = new_values["evaluation"]
			self.send_update()
#			state = self.make_state()
#			if state.result_with_early_stopping() is None:
#				engine_thread.message_queue.put((self, state))

		if (not self.game_over) and self.make_state().result_with_early_stopping() is not None:
			state = self.make_state()
			#scores, final_board = state.compute_scores()
			final_board = state.unconditional_captures()
			self.final_state_to_return = state
			self.player_to_move = state.result_with_early_stopping()
			self.player_move_index = "win"
			self.board = {
				qr: int(final_board.board[qr] + (2 if final_board.board[qr] != self.board[qr] else 0))
				for qr in edgeconnect_rules.ALL_VALID_QR
			}
			self.old_board = self.board.copy()
			self.game_over = True
			self.send_update()

	def send_update(self, from_resume=False):
		# Compute the rendered board.
		rendered_board = {
			qr: v + (4 if self.old_board[qr] != v else 0)
			for qr, v in self.board.items()
		}
		state = self.make_state()
		most_extreme_possible_scores = {
			player: board.compute_scores()[0]
			for player, board in state.make_early_stopping_boards().items()
		}
		most_extreme_possible_scores = {
			player: int(scores[1] - scores[2]) for player, scores in most_extreme_possible_scores.items()
		}
		self.write_message(json.dumps({
			"kind": "board",
			"board": {
				",".join(map(str, qr)): rendered_board[qr]
				for qr in edgeconnect_rules.ALL_VALID_QR
			},
			"aiPlayer": AI_PLAYER_INDEX,
			"boardString": self.make_state().to_string(),
			"isThinking": self.is_thinking,
			"playerToMove": self.player_to_move,
			"playerMoveIndex": self.player_move_index,
			"scoreInterval": most_extreme_possible_scores,
			"nps": self.nps,
			"evaluationCurve": self.evaluation_curve,
			"fromResume": from_resume,
		}))

	def on_close(self):
		print("WebSocket closed.")

def make_app():
	return tornado.web.Application([
		("/api", WebSocketHandler),
	])

if __name__ == "__main__":
	print("Starting engine thread.")
	engine_thread = EngineThread()
	engine_thread.start()
	print("Launching.")
	app = make_app()
	app.listen(8765)
	tornado.ioloop.IOLoop.current().start()
