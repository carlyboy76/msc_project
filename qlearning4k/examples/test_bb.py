#import ipdb
import time

from keras.models import Sequential
from keras.layers import Flatten, Dense
from myGames.bboxgame import Bboxgame
from keras.optimizers import *
from myAgent.agent import Agent
from keras.regularizers import l2, activity_l2

grid_size = 36
nb_frames = 1

def log(message=""):
	fo_log = open("log.txt", "rw+")
	fo_log.seek(0,2)
	fo_log.write(message + "\n")
	fo_log.close()

def run_model(max_moves=1214494, nb_epoch=2000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, \
		lr=0.2, act1='relu', act2='relu', hid1=100, hid2=100, error="mse", reg_param=0.01, id=""):

	#ipdb.set_trace()
	t0 = time.time()
	model = Sequential()
	model.add(Flatten(input_shape=(nb_frames, grid_size)))
	model.add(Dense(hid1, activation=act1, W_regularizer=l2(reg_param), activity_regularizer=activity_l2(reg_param)))
	model.add(Dense(hid2, activation=act2, W_regularizer=l2(reg_param), activity_regularizer=activity_l2(reg_param)))
	model.add(Dense(4, W_regularizer=l2(reg_param), activity_regularizer=activity_l2(reg_param)))
	model.compile(sgd(lr=lr), error)

	print "model summary: ", model.summary()

	game = Bboxgame(max_moves=max_moves)
	agent = Agent(model=model)
	print "training"
	agent.train(game, batch_size=batch_size, nb_epoch=nb_epoch, epsilon=epsilon, epsilon_rate=epsilon_rate, id=id)
	#print "playing"
	#agent.play(game)
	t1 = time.time()

	sec = t1 - t0
	#print "sec: ", str(sec)
	hrs = int(sec / 3600)
	sec -= 3600*hrs
	print "hrs: ", str(hrs),
	#print "sec: ", str(sec)
	mins = int(sec / 60)
	sec -= 60*mins
	print " mins: ", str(mins),
	print " sec: ", str(sec)

	if type(epsilon)  in {tuple, list}:
		log("{:^12}|{:^12}|{:^12.3f}{:^6.3f}{:^6.2f}|{:^10.2f}|{:^20}|{:>3.0f}:{:>02.0f}:{:>02.0f} |{:^6.2f}".format(\
			nb_epoch, batch_size, epsilon[0], epsilon[1], epsilon_rate, lr, act1[:4] + '('+str(hid1)+')' + " + " + act2[:4] + '('+str(hid2)+')', hrs, mins, sec, reg_param))
	else:
		log("{:^12}|{:^12}|{:^12.3f}{:^6.3}{:^6}|{:^10.2f}|{:^20}|{:>3.0f}:{:>02.0f}:{:>02.0f} |{:^6.2f}".format(\
			nb_epoch, batch_size, epsilon, "", "", lr, act1[:4] + '('+str(hid1)+')' + " + " + act2[:4] + '('+str(hid2)+')', hrs, mins, sec, reg_param))


### SCHEDULE RUNS HERE ====================================================================================

log() #spacer
#log("increasing moves - getting NaNs so adding regularisatsion") #spacer

#this is a new job -failed - try again with regularisation below
#log("\n111-init: 10,000 moves")
#run_model(max_moves=10000, nb_epoch=500, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, model_descrip="relu + relu", hidden_size=100, error="mse")
#log("\n111-fini: 10,000 all done\n")

#tried -failed (NaNs suspected)
#log("\n222-init: all moves - let's just see...")
#run_model(nb_epoch=100, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, model_descrip="relu + relu", hidden_size=100, error="mse")
#log("\n222-fini: all runs all done\n")

#this is a new job -done
'''log("\n555-init: 10,000 moves - w/tweaked regularisation + SIGMOIDS (to try to control blow-out)...")
run_model(max_moves=10000, nb_epoch=200, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hidden_size=100, error="mse")
run_model(max_moves=10000, nb_epoch=200, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hidden_size=100, error="mse", reg_param=0.02)
run_model(max_moves=10000, nb_epoch=200, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hidden_size=100, error="mse", reg_param=0.05)
run_model(max_moves=10000, nb_epoch=200, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hidden_size=100, error="mse", reg_param=0.1)
log("\n555-fini: all runs all done\n")'''

#this is a new job -done
'''log("\n333-init: all moves - tweaked regularisation in code (was missing bits - need to understand)...")
run_model(nb_epoch=5, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, model_descrip="relu + relu", hidden_size=100, error="mse")
run_model(nb_epoch=5, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, model_descrip="relu + relu", hidden_size=100, error="mse", reg_param=0.02)
run_model(nb_epoch=5, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, model_descrip="relu + relu", hidden_size=100, error="mse", reg_param=0.05)
run_model(nb_epoch=5, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, model_descrip="relu + relu", hidden_size=100, error="mse", reg_param=0.1)
log("\n333-fini: all runs all done\n")'''

#this is a new job -done
'''log("\n444-init: 10,000 moves - tweaked regularisation in code (was missing bits - need to understand)...")
run_model(max_moves=10000, nb_epoch=200, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, model_descrip="relu + relu", hidden_size=100, error="mse")
run_model(max_moves=10000, nb_epoch=200, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, model_descrip="relu + relu", hidden_size=100, error="mse", reg_param=0.02)
run_model(max_moves=10000, nb_epoch=200, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, model_descrip="relu + relu", hidden_size=100, error="mse", reg_param=0.05)
run_model(max_moves=10000, nb_epoch=200, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, model_descrip="relu + relu", hidden_size=100, error="mse", reg_param=0.1)
log("\n444-fini: all runs all done\n")'''

#this is a new job - done
'''id = str(666)
log("\n"+ id + "-init: 10,000 moves - increasing epochs...")
run_model(max_moves=10000, nb_epoch=200, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hidden_size=100, error="mse", reg_param=0.1)
log(id)
run_model(max_moves=10000, nb_epoch=500, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hidden_size=100, error="mse", reg_param=0.1)
log(id)
run_model(max_moves=10000, nb_epoch=1000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hidden_size=100, error="mse", reg_param=0.1)
log(id)
log("\n"+ id + "-fini: all runs all done\n")'''

#this is a new job - done
'''id = str(888)
log("\n"+ id + "-init: 10,000 moves - increasing batch size...")
run_model(max_moves=10000, nb_epoch=200, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hidden_size=100, error="mse", reg_param=0.1)
log(id)
run_model(max_moves=10000, nb_epoch=200, batch_size=250, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hidden_size=100, error="mse", reg_param=0.1)
log(id)
run_model(max_moves=10000, nb_epoch=200, batch_size=500, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hidden_size=100, error="mse", reg_param=0.1)
log(id)
log("\n"+ id + "-fini: all runs all done\n")'''

#this is a new job - done
'''id = str(222)
log("\n"+ id + "-init: trying fewer units...")
run_model(max_moves=10000, nb_epoch=200, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hidden_size=80, error="mse", reg_param=0.1)
log(id)
run_model(max_moves=10000, nb_epoch=200, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hidden_size=80, error="mse", reg_param=0.1)
log(id)
run_model(max_moves=10000, nb_epoch=200, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hidden_size=50, error="mse", reg_param=0.1)
log(id)
run_model(max_moves=10000, nb_epoch=200, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hidden_size=50, error="mse", reg_param=0.1)
log(id + ' - all done')'''

#this is a new job - running
'''id = str(777)
log("\n"+ id + "-init: all moves - increasing epochs...")
run_model(nb_epoch=10, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id)
run_model(nb_epoch=50, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id)
run_model(nb_epoch=100, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id)
log("\n"+ id + "-fini: all runs all done\n")'''

#this is a new job - running
'''id = str(111)
log("\n"+ id + "-init: trying fewer layers...")
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id)
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id)
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id + ' - all done')'''


#this is a new job - running
'''id = str(222) + ' -dble'
log("\n"+ id + "-init: using relus...")
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id)
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id)
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id + ' - all done')'''


#this is a new job - running
'''id = str(333) + ' -dble'
log("\n"+ id + "-init: full run...") #max_moves=1000,
run_model(nb_epoch=5, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id)
run_model(nb_epoch=5, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id)
run_model(nb_epoch=5, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id + ' - all done')'''

#this is a new job - running
'''id = str(444) + 'a'
log("\n"+ id + "-init: 10k v long run...") #max_moves=1000,
run_model(max_moves=1000, nb_epoch=50000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id + '1')
log(id)
run_model(max_moves=1000, nb_epoch=10000, batch_size=500, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id)
log(id + ' - all done' + '2')'''

#this is a new job - not yet running
id = str(444) + 'b'
log("\n"+ id + "-init: 10k v long run...") #max_moves=1000,
run_model(max_moves=1000, nb_epoch=50000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id + '1')
log(id)
run_model(max_moves=1000, nb_epoch=10000, batch_size=500, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id)
log(id + ' - all done' + '2')


'''id = '4343' #student.compute'
log('quick test run - ' + id)
run_model(max_moves=1000, nb_epoch=10, batch_size=1, epsilon=0.2, epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=100, hid2=100, error="mse", reg_param=0.1, id = id + '1')
log('quick test run done - ' + id)
run_model(max_moves=1000, nb_epoch=10, batch_size=1, epsilon=0.2, epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=100, hid2=100, error="mse", reg_param=0.1, id = id + '2')
log('quick test run done - ' + id)
run_model(max_moves=1000, nb_epoch=10, batch_size=1, epsilon=0.2, epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=100, hid2=100, error="mse", reg_param=0.1, id = id + '3')
log('quick test run done - ' + id)
'''

### =========================================================================================================
