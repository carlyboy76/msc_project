import pickle
import numpy as np

np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(linewidth=250)


scores = pickle.load(open( "score.p", "rb" ))
losses = pickle.load(open( "loss.p", "rb" ))

fo_S = open("score.txt", "rw+")
fo_L = open("loss.txt", "rw+")

fo_S.seek(0,2)
np.savetxt(fo_S, scores, fmt='%4.4f')
fo_L.seek(0,2)
np.savetxt(fo_L, losses, fmt='%4.4f')

fo_S.close()
fo_L.close()


print("scores: ", scores) #(", scores.shape, ") 
print("losses: ", losses) #(", losses.shape, ") 
