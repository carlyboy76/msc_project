import ipdb

from keras.models import Sequential
from keras.layers import Flatten, Dense
from myGames.bboxgame import Bboxgame
from keras.optimizers import *
from qlearning4k import Agent

grid_size = 36
hidden_size = 100
nb_frames = 1

model = Sequential()
model.add(Flatten(input_shape=(nb_frames, grid_size)))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(4))
model.compile(sgd(lr=.2), "mse")

game = Bboxgame()
#ipdb.set_trace()
agent = Agent(model=model)
print "training"
agent.train(game, batch_size=3, nb_epoch=1, epsilon=.1)
print "playing"
agent.play(game)
