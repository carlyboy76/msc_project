__author__ = "Carl Allen"
import datetime as dt
import numpy as np
import interface as bbox
from qlearning4k.games.game import Game
#import ipdb

class Bboxgame(Game):

	def __init__(self, max_moves=1214494):
		#self.grid_size = grid_size
		self.won = False
		self.action_count =0
		self.time = dt.datetime.now()
		self.last_score = 0
		self.action_score = 0
		self.reset()
		self.max_moves=max_moves

	def reset(self):
		#n = np.random.randint(0, self.grid_size-1, size=1)
		#m = np.random.randint(1, self.grid_size-2, size=1)
		if bbox.is_level_loaded():
			bbox.reset_level()
		else:
			bbox.load_level("../../../levels/train_level.data", verbose=1)
		self.state = bbox.get_state() #np.asarray([0, n, m])[np.newaxis]

	@property
	def name(self):
		return "Bb"

	@property
	def nb_actions(self):
		return bbox.get_num_of_actions()

	@property
	def get_action_count(self):
		return self.action_count

	def play(self, action, report_action=False):
		#state = self.state

		printing = False	# SET PRINTING HERE *********************************************************************

		self.action_count = self.action_count + 1
		if report_action and printing:
			print
			print
			print ("PRE ACTN#%d: time=%fs   total score=%f" % (self.action_count, (dt.datetime.now() - self.time).seconds, bbox.get_score()))
			self.time = dt.datetime.now()
		self.has_next = bbox.do_action(action)
		#if action == 0:
		#	action = -1
		#elif action == 1:
		#	action = 0
		#else:
		#	action = 1
		#f0, f1, basket = state[0]
		#new_basket = min(max(1, basket + action), self.grid_size-1)
		#f0 += 1
		#out = np.asarray([f0, f1, new_basket])
		#out = out[np.newaxis]
		#assert len(out.shape) == 2
		#self.state = out

	def get_state(self):
		#im_size = (self.grid_size,) * 2
		#state = self.state[0]
		#canvas = np.zeros(im_size)
		#canvas[state[0], state[1]] = 1
		#canvas[-1, state[2]-1:state[2] + 2] = 1
		return bbox.get_state() #canvas

	def get_score(self):
		#fruit_row, fruit_col, basket = self.state[0]
		#if fruit_row == self.grid_size-1:
		#	if abs(fruit_col - basket) <= 1:
		#		self.won = True
		#		return 1
		#	else:
		#		return -1
		#else:
		#	return 0
		self.action_score = bbox.get_score() - self.last_score
		self.last_score = bbox.get_score()
		return self.action_score #-1 if self.action_score < 0 else (1 if self.action_score > 0 else 0) # min(1, max(0,self.action_score))

	def get_max_moves(self):
		return self.max_moves #100 #1214494

	def get_total_score(self):
		return self.last_score

	def is_over(self):
		#if self.state[0, 0] == self.grid_size-1:
		#	return True
		#else:
		#	return False
		return (self.has_next == 0) or (self.action_count == self.get_max_moves()) # TERMINATION ADDED BY ME *******

	def is_won(self):
		#fruit_row, fruit_col, basket = self.state[0]
		final_score = bbox.get_score()
		bbox.reset_level() # bbox.finish(verbose=1)

		self.last_score = 0
		self.action_count = 0
		return final_score > 0 #fruit_row == self.grid_size-1 and abs(fruit_col - basket) <= 1
