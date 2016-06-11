__author__ = "Carl Allen"

import numpy as np
import interface as bbox
from qlearning4k.games.game import Game


class Bboxgame(Game):

	def __init__(self):
		#self.grid_size = grid_size
		self.won = False
		self.reset()

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

	def play(self, action):
		#state = self.state
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
		return bbox.get_score()

	def is_over(self):
		#if self.state[0, 0] == self.grid_size-1:
		#	return True
		#else:
		#	return False
		return self.has_next == 0

	def is_won(self):
		#fruit_row, fruit_col, basket = self.state[0]
		return self.get_score>0 #fruit_row == self.grid_size-1 and abs(fruit_col - basket) <= 1
