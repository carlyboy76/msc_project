import numpy as np
from random import sample
from keras import backend as K

class Memory:

    def __init__(self):
        pass

    def remember(self, S, a, r, S_prime, game_over):
        pass

    def get_batch(self, model, batch_size):
        pass


class ExperienceReplay(Memory):

    def __init__(self, memory_size=100, fast=None):
        assert not fast or K._BACKEND == 'theano', "Fast mode is avaliable only for theano backend."
        if fast is None and K._BACKEND == 'theano':
            fast = True
        self.fast = fast
        self.memory = []
        self._memory_size = memory_size

    def remember(self, s, a, r, s_prime, game_over, prob):
        self.input_shape = s.shape[1:]
        self.memory.append(np.concatenate([s.flatten(), np.array(a).flatten(), np.array(r).flatten(), s_prime.flatten(), \
		1 * np.array(game_over).flatten(), , np.array(prob).flatten()]))
        if self.memory_size > 0 and len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def get_batch(self, model, batch_size, gamma=0.9): #, print_it=False):
        if self.fast:
            return self.get_batch_fast(model, batch_size, gamma) #, print_it)
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
        nb_actions = model.output_shape[-1]
        samples = np.array(sample(self.memory, batch_size))
        input_dim = np.prod(self.input_shape)
        S = samples[:, 0 : input_dim]
        a = samples[:, input_dim]
        r = samples[:, input_dim + 1]
        S_prime = samples[:, input_dim + 2 : 2 * input_dim + 2]
        game_over = samples[:, 2 * input_dim + 2]
        prob = samples[:, 2 * input_dim + 3]
        r = r.repeat(nb_actions).reshape((batch_size, nb_actions))
        game_over = game_over.repeat(nb_actions).reshape((batch_size, nb_actions))
        probs = prob.repeat(nb_actions).reshape((batch_size, nb_actions))
        S = S.reshape((batch_size, ) + self.input_shape)
        S_prime = S_prime.reshape((batch_size, ) + self.input_shape)
        X = np.concatenate([S, S_prime], axis=0)
        Y = model.predict(X)
        Qsa = np.max(Y[batch_size:], axis=1).repeat(nb_actions).reshape((batch_size, nb_actions))
        delta = np.zeros((batch_size, nb_actions))
        a = np.cast['int'](a)
        delta[np.arange(batch_size), a] = 1
        targets = (1 - delta) * Y[:batch_size] + delta * (r + gamma * (1 - game_over) * Qsa)
        return S, targets, probs

    @property
    def memory_size(self):
        return self._memory_size

    @memory_size.setter
    def memory_size(self, value):
        if value > 0 and value < self._memory_size:
            self.memory = self.memory[:value]
        self._memory_size = value

    def reset_memory(self):
        self.memory = []

    def set_batch_function(self, model, input_shape, batch_size, nb_actions, gamma):
        input_dim = np.prod(input_shape)
        samples = K.placeholder(shape=(batch_size, input_dim * 2 + 3))
        S = samples[:, 0 : input_dim]
        a = samples[:, input_dim]
        r = samples[:, input_dim + 1]
        S_prime = samples[:, input_dim + 2 : 2 * input_dim + 2]
        game_over = samples[:, 2 * input_dim + 2 : 2 * input_dim + 3]
        r = K.reshape(r, (batch_size, 1))
        r = K.repeat(r, nb_actions)
        r = K.reshape(r, (batch_size, nb_actions))
        game_over = K.repeat(game_over, nb_actions)
        game_over = K.reshape(game_over, (batch_size, nb_actions))
        S = K.reshape(S, (batch_size, ) + input_shape)
        S_prime = K.reshape(S_prime, (batch_size, ) + input_shape)
        X = K.concatenate([S, S_prime], axis=0)
        Y = model(X)
        Qsa = K.max(Y[batch_size:], axis=1)
        Qsa = K.reshape(Qsa, (batch_size, 1))
        Qsa = K.repeat(Qsa, nb_actions)
        Qsa = K.reshape(Qsa, (batch_size, nb_actions))
        delta = K.reshape(self.one_hot(a, nb_actions), (batch_size, nb_actions))
        targets = (1 - delta) * Y[:batch_size] + delta * (r + gamma * (1 - game_over) * Qsa)
        self.batch_function = K.function(inputs=[samples], outputs=[S, targets, X, Y, Qsa, delta])

    def  one_hot(self, seq, num_classes):
        import theano.tensor as T
        return K.equal(K.reshape(seq, (-1, 1)), T.arange(num_classes))

    def get_batch_fast(self, model, batch_size, gamma): #, print_it=False):
        if len(self.memory) < batch_size:
            return None
        samples = np.array(sample(self.memory, batch_size))
	'''if print_it:
		print #("[memory] batch_size: %d; memory size: %d" % (batch_size, len(self.memory)))
		#print "samples: ", samples '''
        if not hasattr(self, 'batch_function'):
            self.set_batch_function(model, self.input_shape, batch_size, model.output_shape[-1], gamma)
        S, targets, X, Y, Qsa, delta = self.batch_function([samples])
	#print "[memory] X: "
	#print X
	#print "[memory] Y: ", Y
	#print "[memory] Qsa: ", Qsa
	#print "[memory] delta: ", delta
        return S, targets
