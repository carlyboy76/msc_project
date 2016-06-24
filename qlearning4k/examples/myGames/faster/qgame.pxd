import cython
from cpython cimport bool

cdef class Game:
	
	def __init__(self):
		self.reset()

	#@property
	cdef char* name(self):
		return "Game"
	
	#@property
	cdef int nb_actions(self):
		return 0
	
	cdef reset(self):
		pass

	cdef play(self, action):
		pass

	cdef float* get_state(self):
		#cdef float *array
		return NULL

	cdef float get_score(self):
		return 0

	cdef bool is_over(self):
		return False

	cdef bool is_won(self):
		return False

	cdef float* get_frame(self):
		return self.get_state()

	cdef float* draw(self):
		return self.get_state()
