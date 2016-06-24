import pickle
import numpy as np

np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(linewidth=250)


scores = pickle.load(open( "score.p", "rb" ))
losses = pickle.load(open( "loss.p", "rb" ))

print("scores: ", scores) #(", scores.shape, ") 
print("losses: ", losses) #(", losses.shape, ") 
