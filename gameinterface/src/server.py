#!/usr/bin/python

import sys
sys.path.append("../..")
import engine
import edgeconnect_rules

import json, copy, random, threading, queue, time
import tornado.ioloop
import tornado.web
import tornado.websocket
from tornado import gen

ALL_VALID_QR = [
    (0, 3), (0, 4), (0, 5), (0, 6), (1, 2), (1, 3),
    (1, 4), (1, 5), (1, 6), (2, 1), (2, 2), (2, 3),
    (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2),
    (3, 3), (3, 4), (3, 5), (3, 6), (4, 0), (4, 1),
    (4, 2), (4, 3), (4, 4), (4, 5), (5, 0), (5, 1),
    (5, 2), (5, 3), (5, 4), (6, 0), (6, 1), (6, 2),
    (6, 3),
]

class EngineThread(threading.Thread):
	def __init__(self):
		super().__init__()
		engine.setup_evaluator()
		engine.initialize_model("/home/snp/proj/EdgeConnectZero/run1/models/model-037.npy")
		self.mcts = engine.MCTSEngine()
		self.message_queue = queue.Queue()

	def run(self):
		while True:
			web_handler, state = self.message_queue.get()
			print("Request:", web_handler, state)
			print("To string:", state.to_string())
			self.mcts.set_state(state)
			for _ in range(2):
				move = self.mcts.genmove(2, use_weighted_exponent=5.0)
				print("Got move:", move)
				state.make_move(move)
				self.mcts.set_state(state)
			print("Final:", state.to_string())
			# XXX: Unsynchronized access!
			web_handler.update_queue.put({
				"playerToMove": int(state.move_state[0]),
				"playerMoveIndex": state.move_state[1],
				"board": {qr: int(state.board[qr]) for qr in ALL_VALID_QR},
			})
			#web_handler.player_to_move = 1
			#web_handler.player_move_index = "a"
			#web_handler.board = 
			#web_handler.send_update()

class WebSocketHandler(tornado.websocket.WebSocketHandler):
	def check_origin(self, origin):
		return True

	def open(self):
		print("WebSocket opened to:", self)
		self.player_to_move = 1
		self.player_move_index = "b"
		self.board = {qr: 0 for qr in ALL_VALID_QR}
		self.game_over = False
		self.update_queue = queue.Queue()
		self.send_update()

	def make_state(self):
		state = edgeconnect_rules.EdgeConnectState.initial()
		state.move_state = self.player_to_move, self.player_move_index
		for qr, value in self.board.items():
			state.board[qr] = value
		return state

	def on_message(self, message):
		payload = json.loads(message)
		if payload["kind"] == "click" and not self.game_over:
			qr = tuple(payload["qr"])
			if self.board[qr] != 0:
				return
			# Only let the human play the blue player.
			if self.player_to_move != 1:
				return
			self.board[qr] = self.player_to_move
			if self.player_move_index == "a":
				self.player_move_index = "b"
			else:
				self.player_to_move = 3 - self.player_to_move
				self.player_move_index = "a"
			if self.player_to_move == 2 and self.player_move_index == "a":
				state = self.make_state()
				if state.result() is None:
					engine_thread.message_queue.put((self, state))
			self.send_update()
		if payload["kind"] == "ping":
			self.write_message(json.dumps({"kind": "ping"}))
		if self.update_queue.qsize() and not self.game_over:
			new_values = self.update_queue.get()
			print("Update queue:", new_values)
			self.player_to_move = new_values["playerToMove"]
			self.player_move_index = new_values["playerMoveIndex"]
			self.board = new_values["board"]
			self.send_update()
		if (not self.game_over) and self.make_state().result() is not None:
			state = self.make_state()
			scores, final_board = state.compute_scores()
			self.player_to_move = state.result()
			self.player_move_index = "win"
			self.board = {
				qr: int(final_board.board[qr] + (2 if final_board.board[qr] != self.board[qr] else 0))
				for qr in ALL_VALID_QR
			}
			self.game_over = True
			self.send_update()

	def send_update(self):
		self.write_message(json.dumps({
			"kind": "board",
			"board": {
				",".join(map(str, qr)): self.board[qr]
				for qr in ALL_VALID_QR
			},
			"playerToMove": self.player_to_move,
			"playerMoveIndex": self.player_move_index,
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
