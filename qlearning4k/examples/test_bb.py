#import ipdb
import time

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
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


#this is a new job - not running
'''id = str(555) + ''
log("\n"+ id + "-init: clip relu test...") #max_moves=1000,
run_model(max_moves=100, nb_epoch=100, batch_size=50, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='clipped_relu', act2='clipped_relu', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id + '1c')
log(id)
log("\n"+ id + "-init: clip relu test...") #max_moves=1000,
run_model(max_moves=100, nb_epoch=100, batch_size=50, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id + '1')
log(id + ' - all done')'''


def run_model(max_moves=1214494, nb_epoch=2000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, \
		lr=0.2, act1='relu', max_val1=20, act2='relu', max_val2=20, hid1=100, hid2=100, error="mse", reg_param=0.01, id=""):

	#ipdb.set_trace()
	t0 = time.time()
	model = Sequential()
	model.add(Flatten(input_shape=(nb_frames, grid_size)))
	#model.add(Dense(hid1, activation=act1, W_regularizer=l2(reg_param), activity_regularizer=activity_l2(reg_param)))
	#model.add(Dense(hid2, activation=act2, W_regularizer=l2(reg_param), activity_regularizer=activity_l2(reg_param)))
	model.add(Dense(hid1, W_regularizer=l2(reg_param), activity_regularizer=activity_l2(reg_param)))
	if act1 == 'clip':
		model.add(Activation('relu', max_value=max_val1))
	else: 
		model.add(Activation(act1))
	model.add(Dense(hid2, W_regularizer=l2(reg_param), activity_regularizer=activity_l2(reg_param)))
	if act2 == 'clip':
		model.add(Activation('relu', max_value=max_val2))
	else: 
		model.add(Activation(act2))
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
			nb_epoch, batch_size, epsilon[0], epsilon[1], epsilon_rate, lr, \
			act1[:4] + '('+str(hid1)+')' + " + " + act2[:4] + '('+str(hid2)+')', \
			hrs, mins, sec, reg_param))
	else:
		log("{:^12}|{:^12}|{:^12.3f}{:^6.3}{:^6}|{:^10.2f}|{:^20}|{:>3.0f}:{:>02.0f}:{:>02.0f} |{:^6.2f}".format(\
			nb_epoch, batch_size, epsilon, "", "", lr, \
			act1[:4] + '('+str(hid1)+')' + " + " + act2[:4] + '('+str(hid2)+')', \
			hrs, mins, sec, reg_param))


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

#this is a new job - done
'''id = str(111)
log("\n"+ id + "-init: trying fewer layers...")
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id)
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id)
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id + ' - all done')'''


#this is a new job - done
'''id = str(222) + ' -dble'
log("\n"+ id + "-init: using relus...")
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id)
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id)
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id + ' - all done')'''


#this is a new job - done
'''id = str(333) + ' -dble'
log("\n"+ id + "-init: full run...") #max_moves=1000,
run_model(nb_epoch=5, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id)
run_model(nb_epoch=5, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1)
log(id)
run_model(nb_epoch=5, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1)
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
'''id = str(444) + 'a'
log("\n"+ id + "-init: 10k v long run...") #max_moves=1000,
run_model(max_moves=1000, nb_epoch=50000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id + '1')
log(id)
run_model(max_moves=1000, nb_epoch=50000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id + '1')
'''

#this is a new job - running
'''id = str(444) + 'b'
log("\n"+ id + "-init: 10k v long run...") #max_moves=1000,
run_model(max_moves=1000, nb_epoch=10000, batch_size=500, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id)
log(id + ' - all done' + '2')log(id)
run_model(max_moves=1000, nb_epoch=10000, batch_size=500, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id)
log(id + ' - all done' + '2')'''

#this is a new job - running - THESE WIL OVERWRITE SO NEED TO COPY
'''id = str(555)
log("\n"+ id + "-init: 50k - replicate 10k...") #max_moves=1000,
run_model(max_moves=1000, nb_epoch=50000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95/5, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id + '1')
log(id)
run_model(max_moves=1000, nb_epoch=50000, batch_size=100, epsilon=[1, 0.1], epsilon_rate=0.95/5, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id + '1')
log(id)
log(id + ' - all done')'''

#this is a new job - running - THESE WIL OVERWRITE SO NEED TO COPY
'''id = str(666)
log("\n"+ id + "-init: 50k - eps to zero...") #max_moves=1000,
run_model(max_moves=1000, nb_epoch=50000, batch_size=100, epsilon=[1, 0], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'1')
run_model(max_moves=1000, nb_epoch=50000, batch_size=100, epsilon=[1, 0], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'2')
log(id + ' - all done')'''

#this is a new job - running
'''id = str(666) + 'b'
log("\n"+ id + "-init: 50k - eps to near zero...") #max_moves=1000,
run_model(max_moves=1000, nb_epoch=50000, batch_size=100, epsilon=[1, 0.0001], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'1')
run_model(max_moves=1000, nb_epoch=50000, batch_size=100, epsilon=[1, 0.0001], epsilon_rate=0.95, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'2')
log(id + ' - all done')'''

#this is a new job - running
'''id = str(666) +'r'
log("\n"+ id + "-init: 50k - eps to zero...") #max_moves=1000,
run_model(max_moves=1000, nb_epoch=50000, batch_size=100, epsilon=[1, 0], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'1')
run_model(max_moves=1000, nb_epoch=50000, batch_size=100, epsilon=[1, 0], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'2')
log(id + ' - all done')'''

#this is a new job - running
'''id = str(1111)
log("\n"+ id + "-init: 10k - eps 0.2 -> 0.0001 ...") 
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[0.2, 0.0001], epsilon_rate=0.90, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_1')
log(id)
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[0.2, 0.0001], epsilon_rate=0.90, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_2')
log(id)
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[0.2, 0.0001], epsilon_rate=0.90, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_3')
log(id + ' - all done')'''

#this is a new job - running
'''id = str(1112)
log("\n"+ id + "-init: 10k - eps 0.5 -> 0.0001 ...") 
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[0.5, 0.0001], epsilon_rate=0.90, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_1')
log(id)
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[0.5, 0.0001], epsilon_rate=0.90, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_2')
log(id)
run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[0.5, 0.0001], epsilon_rate=0.90, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_3')
log(id + ' - all done')'''

#this is a new job - running
'''id = str(1113)
log("\n"+ id + "-init: minitest ...") 
run_model(max_moves=100, nb_epoch=100, batch_size=100, epsilon=[0.5, 0.0001], epsilon_rate=0.90, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_1')'''
'''log(id)
run_model(max_moves=1000, nb_epoch=100, batch_size=100, epsilon=[0.5, 0.0001], epsilon_rate=0.90, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_2')
log(id)
run_model(max_moves=1000, nb_epoch=100, batch_size=100, epsilon=[0.5, 0.0001], epsilon_rate=0.90, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_3')
log(id + ' - all done')'''

job=9
id = str(1114)
if job == 1:
	log("\n"+ id + "-init:  eps") 
	run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0], epsilon_rate=0.70, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_1')
	log(id)
elif job == 2:
	run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0], epsilon_rate=0.80, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_2')
	log(id)
elif job == 3:
	run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0], epsilon_rate=0.90, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_3')
	log(id + ' - all done')

#this is a new job - running
id = str(1115)
if job == 4:
	log("\n"+ id + "-init:  lr") 
	run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0], epsilon_rate=0.80, lr=0.1, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_1')
	log(id)
elif job == 5:
	run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0], epsilon_rate=0.80, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_2')
	log(id)
elif job == 6:
	run_model(max_moves=1000, nb_epoch=10000, batch_size=100, epsilon=[1, 0], epsilon_rate=0.80, lr=0.3, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_3')
	log(id + ' - all done')

#this is a new job - running
id = str(1116)
if job == 7:
	log("\n"+ id + "-init:  bs=200") 
	run_model(max_moves=1000, nb_epoch=5000, batch_size=200, epsilon=[1, 0], epsilon_rate=0.80, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_1')
	log(id + ' - all done')

#this is a new job - running
id = str(1117)
if job == 8:
	log("\n"+ id + "-init:  bs=400") 
	run_model(max_moves=1000, nb_epoch=2500, batch_size=400, epsilon=[1, 0], epsilon_rate=0.80, lr=0.2, act1='sigmoid', act2='sigmoid', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_1')
	log(id + ' - all done')

id = str(9999)
if job == 9:
	log("\n"+ id + "-init:  ruql??") 
	run_model(max_moves=100, nb_epoch=100, batch_size=100, epsilon=[1, 0.01], epsilon_rate=0.80, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'_1')
	log(id + ' - all done')




############## TRYING TO CLIP RELU  ###########################################


#this is a new job - NOT running
'''id = str(9999) +'rc'  # trying to clip relus
log("\n"+ id + "-init: quick test - relu clipped...") #max_moves=1000,
run_model(max_moves=100, nb_epoch=100, batch_size=50, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='clip', max_val1=20, act2='clip', max_val2=20, hid1=50, hid2=50, error="mse", reg_param=0.1, id = id+'1')
log(id + ' - all done')'''


#this is a new job - not running
'''id = str(555) + ''
log("\n"+ id + "-init: clip relu test...") #max_moves=1000,
run_model(max_moves=100, nb_epoch=100, batch_size=50, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='clipped_relu', act2='clipped_relu', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id + '1c')
log(id)
log("\n"+ id + "-init: clip relu test...") #max_moves=1000,
run_model(max_moves=100, nb_epoch=100, batch_size=50, epsilon=[1, 0.1], epsilon_rate=0.95, lr=0.2, act1='relu', act2='relu', hid1=50, hid2=50, error="mse", reg_param=0.1, id = id + '1')
log(id + ' - all done')'''


### =========================================================================================================
