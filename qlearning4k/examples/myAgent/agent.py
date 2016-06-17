from memory import ExperienceReplay
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import ipdb
import pickle

np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(linewidth=250)

class Agent:

	def __init__(self, model, memory=None, memory_size=1000, nb_frames=None):
		assert len(model.output_shape) == 2, "Model's output shape should be (nb_samples, nb_actions)."
		if memory:
			self.memory = memory
		else:
			self.memory = ExperienceReplay(memory_size)
		if not nb_frames and not model.input_shape:
			raise Exception("Missing argument : nb_frames not provided")
		elif not nb_frames:
			nb_frames = model.input_shape[1]
		elif model.input_shape[1] and nb_frames and model.input_shape[1] != nb_frames:
			raise Exception("Dimension mismatch : time dimension of model should be equal to nb_frames.")
		self.model = model
		self.nb_frames = nb_frames
		self.frames = None

	@property
	def memory_size(self):
		return self.memory.memory_size

	@memory_size.setter
	def memory_size(self, value):
		self.memory.memory_size = value

	def reset_memory(self):
		self.exp_replay.reset_memory()

	def check_game_compatibility(self, game):
		game_output_shape = (1, None) + game.get_frame().shape
		if len(game_output_shape) != len(self.model.input_shape):
			raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
		else:
			for i in range(len(self.model.input_shape)):
				if self.model.input_shape[i] and game_output_shape[i] and self.model.input_shape[i] != game_output_shape[i]:
					raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
		if len(self.model.output_shape) != 2 or self.model.output_shape[1] != game.nb_actions:
			raise Exception('Output shape of model should be (nb_samples, nb_actions).')

	def get_game_data(self, game):
		frame = game.get_frame()
		if self.frames is None:
			self.frames = [frame] * self.nb_frames
		else:
			self.frames.append(frame)
			self.frames.pop(0)
		return np.expand_dims(self.frames, 0)

	def clear_frames(self):
		self.frames = None

	def action_count(self, game):
		return game.get_action_count

	# SET WHICH RUNS TO PRINT OUT HERE *****************************************************************
	def report_action(self, game):
		self.report_freq = 100000
		return ((self.action_count(game) % self.report_freq) >= 0) and ((self.action_count(game) % self.report_freq) < 20) #% 10000) == 0 #

	def train(self, game, nb_epoch=1000, batch_size=50, gamma=0.9, epsilon=[1., .1], epsilon_rate=0.5, reset_memory=False):
		fo_A = open("A.txt", "rw+")
		fo_G = open("G.txt", "rw+")
		fo_Gb = open("Gb.txt", "rw+")
		fo_I = open("I.txt", "rw+")
		fo_Q = open("Q.txt", "rw+")
		fo_R = open("R.txt", "rw+")
		fo_S = open("S.txt", "rw+")
		fo_T = open("T.txt", "rw+")
		fo_W = open("W.txt", "rw+")
		fo_Wb = open("Wb.txt", "rw+")

		self.check_game_compatibility(game)
		if type(epsilon)  in {tuple, list}:
			delta =  ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))
			final_epsilon = epsilon[1]
			epsilon = epsilon[0]
		else:
			final_epsilon = epsilon
		model = self.model
		nb_actions = model.output_shape[-1]
		win_count = 0

		scores = [] #np.zeros((nb_epoch,5/self.report_freq+1))
		losses = [] #np.zeros((nb_epoch,5/self.report_freq+1))


		for epoch in range(nb_epoch):
			#ipdb.set_trace(context=9)	# TRACING HERE *********************************************
			loss = 0.
			game.reset()
			self.clear_frames()
			if reset_memory:
				self.reset_memory()
			game_over = False
			S = self.get_game_data(game)
			no_last_S = True

			plot_showing = False

			while not game_over:
				if np.random.random() < epsilon:
					a = int(np.random.randint(game.nb_actions))
					#if (self.action_count(game) % 100000) == 0:
					if self.report_action(game):
						print "random",
						q = model.predict(S)
				else:
					q = model.predict(S)
					#print q.shape
					#print q[0]
					if (q[0,0] != q[0,0]):
						ipdb.set_trace(context=9)	# TRACING HERE *********************************************
						
					a = int(np.argmax(q[0]))
					#if (self.action_count(game) % 100000) == 0:
				game.play(a, self.report_action(game))
				r = game.get_score()


				# PRINTING S HERE ******************************************************************

				''' if plot_showing:	
					plt.clf()
				plt.imshow(np.reshape(S,(6,6)))
				plt.draw()
				plt.show(block=False)
				plot_showing = True
				print "hi" '''

				# PRINTING S HERE ******************************************************************

				S_prime = self.get_game_data(game)
				


				if self.report_action(game):
					print "S: ", S
					if no_last_S:
						last_S = S
						no_last_S = False
					else:
						print "dS:", S - last_S
						print "    ==>  Q(lS):", model.predict(last_S)
					print
					print "    ==>  Q(S): ", q, "    ==>  A: ", a, "    ==> R: %f" % r
					print "    ==>  Q(S'):", model.predict(S_prime)
					#print
					fo_S.seek(0,2)
					np.savetxt(fo_S, S[0], fmt='%4.4f') #
					fo_Q.seek(0,2)
					np.savetxt(fo_Q, q, fmt='%4.4f') #
					fo_A.seek(0,2)
					fo_A.write(str(a)+"\n") #savetxt(fo, S[0], fmt='%4.4f') #
					fo_R.seek(0,2)
					fo_R.write(str(r)+"\n")


				#ipdb.set_trace(context=9)	# TRACING HERE *********************************************


				last_S = S

				game_over = game.is_over()
				transition = [S, a, r, S_prime, game_over]
				self.memory.remember(*transition)
				S = S_prime
				batch = self.memory.get_batch(model=model, batch_size=batch_size, gamma=gamma, \
					print_it=self.report_action(game))
				if batch:
					inputs, targets = batch

					#print("model.total_loss: ", model.total_loss)

					weights_pre = model.get_weights() # GOT WEIGHTS *************************
					#print "weights_pre"
					#print weights_pre

					if self.report_action(game):
						fo_W.seek(0,2)
						np.savetxt(fo_W, weights_pre[0], fmt='%4.4f') #
						fo_W.write("\n")
						fo_Wb.seek(0,2)
						np.savetxt(fo_Wb, weights_pre[1], fmt='%4.4f') #
						fo_Wb.write("\n")

					output = model.train_on_batch(inputs, targets)
					loss += float(output[0]) #model.train_on_batch(inputs, targets))
					

					if self.report_action(game):
						#print output
						fo_G.seek(0,2)
						np.savetxt(fo_G, output[1], fmt='%4.4f') #
						fo_G.write("\n")
						fo_Gb.seek(0,2)
						np.savetxt(fo_Gb, output[2], fmt='%4.4f') #
						fo_Gb.write("\n")

					#weights_post = model.get_weights() # GOT WEIGHTS ********************************
					#print "weights_post"
					#print weights_post
					#ipdb.set_trace()	# TRACING HERE *********************************************

					if self.report_action(game):
						action_count = self.action_count(game)
						scores.append(game.get_total_score()) #[epoch,action_count/self.report_freq] = game.get_total_score()
						losses.append(loss) # 	[epoch,action_count/self.report_freq] = loss

						#print ("running a batch (of %d): 1: %d; 2: %d" % (len(batch), batch[0].size, \
						#	batch[1].size))
						#print "memory size: ", self.memory_size
						#print "using memory\n", inputs, "; tgt: ", targets
						fo_I.seek(0,2)
						np.savetxt(fo_I, inputs[0], fmt='%4.4f') #
						fo_T.seek(0,2)
						np.savetxt(fo_T, targets, fmt='%4.4f') #
					#fo_T.write("\n")
			if game.is_won():
				win_count += 1
			if epsilon > final_epsilon:
				epsilon -= delta
			print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Win count {}".format(epoch + 1, nb_epoch, loss, epsilon, win_count))
			fo_A.close()
			fo_G.close()
			fo_Gb.close()
			fo_I.close()
			fo_Q.close()
			fo_R.close()
			fo_S.close()
			fo_T.close()
			fo_W.close()
			fo_Wb.close()

		#pickle.dump(scores, open( "score.p", "wb" ) )
		#pickle.dump(losses, open( "loss.p", "wb" ) )

	def play(self, game, nb_epoch=1, epsilon=0., visualize=False):
		self.check_game_compatibility(game)
		model = self.model
		win_count = 0
		frames = []
		for epoch in range(nb_epoch):
			game.reset()
			self.clear_frames()
			S = self.get_game_data(game)
			if visualize:
				frames.append(game.draw())
			game_over = False
			while not game_over:
				if np.random.rand() < epsilon:
					print("random")
					action = int(np.random.randint(0, game.nb_actions))
				else:
					q = model.predict(S)			
					action = int(np.argmax(q[0]))
				game.play(action)
				S = self.get_game_data(game)
				if visualize:
					frames.append(game.draw())
				game_over = game.is_over()
			if game.is_won():
				win_count += 1
		print("Accuracy {} %".format(100. * win_count / nb_epoch))
		if visualize:
			if 'images' not in os.listdir('.'):
				os.mkdir('images')
			for i in range(len(frames)):
				plt.imshow(frames[i], interpolation='none')
				plt.savefig("images/" + game.name + str(i) + ".png")
 
