import random
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib import colors

GRID_SIZE=30
time_steps = 25
random.seed(300)

filtering_arg=int(sys.argv[1])
smoothing_arg=int(sys.argv[2])
manhattan_arg=int(sys.argv[3])
prediction_arg=int(sys.argv[4])
mostlikely_arg=int(sys.argv[5])



class State():
	def __init__(self, row, col):
		self.row=row
		self.col=col
	
	def valid(self):
		if 0<=self.row<GRID_SIZE and 0<=self.col<GRID_SIZE:
			return True
		else:
			return False
	
	def corner(self):
		if self.row==GRID_SIZE-1 and (self.col==0 or self.col==GRID_SIZE-1):
			return True
		elif self.row==0 and (self.col==0 or self.col==GRID_SIZE-1):
			return True
		else:
			return False
	
	def edge(self):
		if self.row==GRID_SIZE-1 and 0<self.col<GRID_SIZE-1:
			return True
		elif self.col==GRID_SIZE-1 and  0<self.row<GRID_SIZE-1:
			return True
		elif self.row==0 and 0<self.col<GRID_SIZE-1:
			return True
		elif self.col==0 and 0<self.row<GRID_SIZE-1:
			return True
		else:
			return False


def find_state(row, col):
	return ((GRID_SIZE*row) + col)

def find_coordinates(state):
	return State(state//GRID_SIZE, state%GRID_SIZE)

def transition_matrix():

	T = np.zeros((GRID_SIZE**2, GRID_SIZE**2))
	for i in range(GRID_SIZE):
		for j in range(GRID_SIZE):
			index = find_state(i, j)
			state = State(i,j)
			if state.edge():
				if i==0:
					T[index][find_state(i,j-1)] = 0.2
					T[index][find_state(i,j+1)] = 0.3
					T[index][index] = 0.4
					T[index][find_state(i+1,j)] = 0.1
				elif i==GRID_SIZE-1:
					T[index][find_state(i,j-1)] = 0.2
					T[index][index] = 0.1
					T[index][find_state(i,j+1)] = 0.3
					T[index][find_state(i-1,j)] = 0.4
				elif j==0:
					T[index][find_state(i-1,j)] = 0.4
					T[index][find_state(i+1,j)] = 0.1
					T[index][find_state(i,j+1)] = 0.3
					T[index][index] = 0.2
				else:
					T[index][find_state(i-1,j)] = 0.4
					T[index][find_state(i+1,j)] = 0.1
					T[index][index] = 0.3
					T[index][find_state(i,j-1)] = 0.2
			elif state.corner():
				if i==0 and j==0:
					T[index][find_state(i,j+1)] = 0.3
					T[index][find_state(i+1,j)] = 0.1
					T[index][index] = 0.6
				elif i==0 and j==GRID_SIZE-1:
					T[index][index] = 0.7
					T[index][find_state(i,j-1)] = 0.2
					T[index][find_state(i+1,j)] = 0.1
				elif i==GRID_SIZE-1 and j==0:
					T[index][find_state(i,j+1)] = 0.3
					T[index][index] = 0.3
					T[index][find_state(i-1,j)] = 0.4
				else:
					T[index][find_state(i,j-1)] = 0.2
					T[index][index] = 0.4
					T[index][find_state(i-1,j)] = 0.4
			else:
				T[index][find_state(i,j+1)] = 0.3
				T[index][find_state(i,j-1)] = 0.2
				T[index][find_state(i+1,j)] = 0.1
				T[index][find_state(i-1,j)] = 0.4
				

	return T

T=transition_matrix()

class Sensor():
	def __init__(self):
		self.SENSOR1 = 0
		self.SENSOR2 = 1
		self.SENSOR3 = 2
		self.SENSOR4 = 3

class Observation_Model():
	def __init__(self):
		self.sensor=np.array([[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],
							[0.5,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5],
							[0.5,0.6,0.7,0.7,0.7,0.7,0.7,0.6,0.5],
							[0.5,0.6,0.7,0.8,0.8,0.8,0.7,0.6,0.5],
							[0.5,0.6,0.7,0.8,0.9,0.8,0.7,0.6,0.5],
							[0.5,0.6,0.7,0.8,0.8,0.8,0.7,0.6,0.5],
							[0.5,0.6,0.7,0.7,0.7,0.7,0.7,0.6,0.5],
							[0.5,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5],
							[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])
		self.sensor1=np.zeros((GRID_SIZE,GRID_SIZE))
		self.sensor1[10:19,3:12]=self.sensor
		self.sensor2=np.zeros((GRID_SIZE,GRID_SIZE))
		self.sensor2[10:19,10:19]=self.sensor
		self.sensor3=np.zeros((GRID_SIZE,GRID_SIZE))
		self.sensor3[10:19,17:26]=self.sensor
		self.sensor4=np.zeros((GRID_SIZE,GRID_SIZE))
		self.sensor4[3:12,10:19]=self.sensor

	def sample(self, state, obs):
		prob=0
		Sensor1=Sensor()
		if obs==Sensor1.SENSOR1:
			prob=self.sensor1[state.row,state.col]
		elif obs==Sensor1.SENSOR2:
			prob=self.sensor2[state.row,state.col]
		elif obs==Sensor1.SENSOR3:
			prob=self.sensor3[state.row,state.col]
		elif obs==Sensor1.SENSOR4:
			prob=self.sensor4[state.row,state.col]

		measure = np.random.choice([True,False], 1, p = [prob, 1.0-prob])[0]
		# print(measure)
		return measure

	def observation_matrix(self):
		sensor1_emission=self.sensor1.flatten()
		sensor2_emission=self.sensor2.flatten()
		sensor3_emission=self.sensor3.flatten()
		sensor4_emission=self.sensor4.flatten()
		
		# s1 not (s2,s3,s4)
		obs_zero=sensor1_emission*(1-sensor2_emission)*(1-sensor3_emission)*(1-sensor4_emission)
		# s2 not (s1,s3,s4)
		obs_one=sensor2_emission*(1-sensor1_emission)*(1-sensor3_emission)*(1-sensor4_emission)
		# s3 not (s1,s2,s4)
		obs_two=sensor3_emission*(1-sensor1_emission)*(1-sensor2_emission)*(1-sensor4_emission)
		# s4 not (s1,s2,s3)
		obs_three=sensor4_emission*(1-sensor1_emission)*(1-sensor2_emission)*(1-sensor3_emission)
		# s1, s2 not (s3,s4)
		obs_four=sensor1_emission*sensor2_emission*(1-sensor3_emission)*(1-sensor4_emission)
		# s3, s2 not (s1,s4)
		obs_five=sensor3_emission*sensor2_emission*(1-sensor1_emission)*(1-sensor4_emission)
		# s4, s2 not (s1,s3)
		obs_six=sensor4_emission*sensor2_emission*(1-sensor3_emission)*(1-sensor1_emission)
		# s1,s2,s4 not s3
		obs_seven=sensor1_emission*sensor2_emission*sensor4_emission*(1-sensor3_emission)
		# s3,s2,s4 not s3
		obs_eight=sensor3_emission*sensor2_emission*sensor4_emission*(1-sensor1_emission)
		# not (s1,s2,s3,s4)
		obs_nine=(1-sensor1_emission)*(1-sensor2_emission)*(1-sensor3_emission)*(1-sensor4_emission)

		return np.vstack((obs_zero, obs_one, obs_two, obs_three,obs_four, obs_five, obs_six, obs_seven,obs_eight,obs_nine)).T


def prior_probability():
	return float(1/(GRID_SIZE*GRID_SIZE))*np.ones((GRID_SIZE, GRID_SIZE)).flatten()

P_init = prior_probability()

class Environment_Model():
	def __init__(self):
		self.obs = Observation_Model()
		self.init_state = State(14,14)
	
	
	def get_state_from_action(self, state, action):
		next_state_map = {0: State(state.row-1, state.col),
						   1: State(state.row+1, state.col),
						   2: State(state.row, state.col-1),
						   3: State(state.row, state.col+1)}
		next_state = next_state_map[action]
		if next_state.valid():
			return next_state
		else:
			return state

	def get_observation(self, state):
		Sensor1=Sensor()
		s1=self.obs.sample(state,Sensor1.SENSOR1)
		s2=self.obs.sample(state,Sensor1.SENSOR2)
		s3=self.obs.sample(state,Sensor1.SENSOR3)
		s4=self.obs.sample(state,Sensor1.SENSOR4)

		if s1 and (not s2) and (not s3) and (not s4):
			return 0

		elif s2 and (not s1) and (not s3) and (not s4):
			return 1

		elif s3 and (not s2) and (not s1) and (not s4):
			return 2

		elif s4 and (not s2) and (not s3) and (not s1):
			return 3

		elif s1 and s2 and (not s3) and (not s4):
			return 4

		elif s3 and s2 and (not s1) and (not s4):
			return 5

		elif s4 and s2 and (not s3) and (not s1):
			return 6

		elif s1 and s2 and s4 and (not s3):
			return 7

		elif s3 and s2 and s4 and (not s1):
			return 8

		else:
			return 9

	def sim_step(self, state):
		x_act=np.random.choice([0,1,2,3],p=[0.4,0.1,0.2,0.3])
		next_state = self.get_state_from_action(state, x_act)
		observation = self.get_observation(next_state)
		return next_state, x_act, observation

def simulate1(env,time_steps):
	states,actions,observed = [env.init_state],[],[env.get_observation(env.init_state)]



	for time in range(time_steps-1):
		state, action, obs = env.sim_step(env.init_state)
		actions.append(action)
		states.append(state)
		observed.append(obs)
		# print(action)
	return states, actions, observed

def constant_norm(data):
	alpha = data.sum(axis=1)
	data = data/alpha[:, np.newaxis] 
	return data

ENVIRONMENT = Environment_Model()
O = ENVIRONMENT.obs.observation_matrix()

states, actions, OBS = simulate1(ENVIRONMENT,time_steps)

for s in states:
	print(s.row,s.col)

def filtering(OBS, P_init, O, T):
	likelihood = np.zeros((len(OBS), len(P_init)))
	likelihood[0,:] = P_init * O[:, OBS[0]] 
	for i in range(1,len(OBS)):
		for j in range(len(P_init)):
			likelihood[i, j] = np.matmul(T[:,j], likelihood[i-1, :]) * O[j, OBS[i]]
	likelihood = constant_norm(likelihood)
	return likelihood

alpha = filtering(OBS, P_init, O, T)

def plot_loglikelihood(dataa, title):
	fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(18,18))
	fig.suptitle(title)
	plt.subplots_adjust(left=0.01, bottom=None, right=0.99, top=None, wspace=0.40, hspace=None)
	t = 0
	for ax in axes.flat:
		ax.set_title('t='+str(t),fontsize=8)
		im = ax.imshow(np.log(dataa[t].reshape(GRID_SIZE,GRID_SIZE)), cmap='bone', origin='upper')
		ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.3)
		ax.set_xticks(np.arange(0, 30, 1))
		ax.set_yticks(np.arange(0, 30, 1))
		ax.tick_params(axis='x', colors='white')
		ax.tick_params(axis='y', colors='white')
		t=t+ 1
	fig.colorbar(im, ax=axes.ravel().tolist())
	plt.show()

if filtering_arg:
	plot_loglikelihood(alpha, "Log Likelihood - Filtering")

def plot_groundtruth_prediction(dataa, title):
	fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(28,28))
	fig.suptitle(title)
	plt.subplots_adjust(left=0.01, bottom=None, right=0.99, top=None, wspace=0.40, hspace=None)
	t = 0
	pred_states=[]
	for ax in axes.flat:
		ax.set_title('t='+str(t),fontsize=8)
		ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.3)
		ax.set_xticks(np.arange(0, 30, 1))
		ax.set_yticks(np.arange(0, 30, 1))
		ax.tick_params(axis='x', colors='white')
		ax.tick_params(axis='y', colors='white')
		index = np.argmax(dataa[t])
		pred_state = find_coordinates(index)
		pred_states.append(pred_state)
		true_state = states[t]
		data = np.zeros((GRID_SIZE, GRID_SIZE))
		if pred_state.row==true_state.row and pred_state.col==true_state.col:
			data[true_state.row, true_state.col] = -0.20
		else:
			data[pred_state.row, pred_state.col] = -0.85
			data[true_state.row, true_state.col] = -1.0
		im = ax.imshow(data, cmap='pink', origin='upper')
		t+=1
	fig.colorbar(im, ax=axes.ravel().tolist())
	plt.show()
	return np.array(pred_states)

if filtering_arg:
	pred_states1=plot_groundtruth_prediction(alpha, "Predicted v/s Ground Truth - Filtering")

def recompute(OBS, P_init, O, T):
	back = np.zeros((len(OBS), len(P_init)))
	back[len(OBS)-1] = np.ones((len(P_init)))
	for i in range(len(OBS)-2,-1,-1):
		for j in range(len(P_init)):
			back[i, j] = np.matmul((back[i+1, :] * O[:, OBS[i+1]]),(T[j,:]))
	back = constant_norm(back)
	return back

def smoothing(OBS, P_init, O, T):
	front = filtering(OBS, P_init, O, T)
	back = recompute(OBS, P_init, O, T)
	smooth_dist = front*back
	smooth_dist = constant_norm(smooth_dist)
	return smooth_dist


smoothing_dist = smoothing(OBS, P_init, O, T)

if smoothing_arg:
	pred_states2=plot_groundtruth_prediction(smoothing_dist, "Predicted v/s Ground Truth - Smoothing")

filt_last_likelihood = alpha[-1]

def predict_future(filt_last_likelihood, T, steps):
	likelihoods = []
	for t in range(steps):
		filt_last_likelihood = np.matmul(T, filt_last_likelihood)
		likelihoods.append(filt_last_likelihood)
	likelihoods = [np.log(x.reshape(GRID_SIZE, GRID_SIZE)) for x in likelihoods]
	return likelihoods


def plot_future(likelihoods,rows):
	fig, axes = plt.subplots(nrows=rows, ncols=5, figsize=(18,18))
	fig.suptitle("Predictive Dist. over future location for " +str(5*rows)+ " time steps")
	plt.subplots_adjust(left=0.01, bottom=None, right=0.99, top=None, wspace=0.40, hspace=None)
	t = 0
	# print(len)
	for ax in axes.flat:
		ax.set_title('t='+str(t+25),fontsize=8)
		ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.3)
		ax.set_xticks(np.arange(0, 30, 1))
		ax.set_yticks(np.arange(0, 30, 1))
		ax.tick_params(axis='x', colors='white')
		ax.tick_params(axis='y', colors='white')
		im = ax.imshow(likelihoods[t], cmap='bone', origin='upper')
		t+=1
	fig.colorbar(im, ax=axes.ravel().tolist())
	plt.show()

if prediction_arg:
	likelihood_10 = predict_future(filt_last_likelihood, T,10)

	plot_future(likelihood_10,2)

	likelihood_25=predict_future(filt_last_likelihood, T,25)

	plot_future(likelihood_25,5)


def most_likely_path(OBS, P_init, O, T):
	temp = np.zeros((len(OBS), len(P_init)))
	temp[0, :] = np.log(P_init * O[:, OBS[0]])
	before = np.zeros((len(OBS)-1, len(P_init)))
	for i in range(1,len(OBS)):
		for j in range(len(P_init)):
			prob_dist = temp[i-1] + np.log(T[:, j]*O[j, OBS[i]])
			before[i-1, j] = np.argmax(prob_dist)
			temp[i, j] = np.max(prob_dist)
	path = []
	maxarg = np.argmax(temp[len(OBS)-1, :])
	path.append(maxarg)
	for k in range(len(OBS)-2,-1,-1):
		path.append(before[k,int(maxarg)])
		maxarg = before[k,int(maxarg)]
	path=np.flip(path).astype(np.uint)
	return path

def plot_mostlikelypath(path, title):
	fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(18,18))
	fig.suptitle(title)
	plt.subplots_adjust(left=0.01, bottom=None, right=0.99, top=None, wspace=0.40, hspace=None)
	t = 0
	for ax in axes.flat:
		ax.set_title('t='+str(t),fontsize=8)
		ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.3)
		ax.set_xticks(np.arange(0, 30, 1))
		ax.set_yticks(np.arange(0, 30, 1))
		ax.tick_params(axis='x', colors='white')
		ax.tick_params(axis='y', colors='white')
		data = np.zeros((GRID_SIZE, GRID_SIZE))
		p = (path[t]//GRID_SIZE, path[t]%GRID_SIZE)
		data[p[0], p[1]] = -0.85
		im = ax.imshow(data, cmap='pink', origin='upper')
		t+=1
	fig.colorbar(im, ax=axes.ravel().tolist())
	plt.show()

if mostlikely_arg:
	path = most_likely_path(OBS, P_init, O, T)
	print("most_likely_path: ", path)
	plot_mostlikelypath(path,  "Most Likely Path")

def compute_manhattan_error(pred, actual,task):
	dist = []
	for i in range(len(pred)):
		p = (pred[i].row, pred[i].col)
		o = (actual[i].row, actual[i].col)
		dist.append(abs(p[0]-o[0]) + abs(p[1]-o[1]))

	plt.figure(figsize=(8, 8))
	plt.plot(np.arange(time_steps), dist, marker='*')
	plt.title("Error between estimated and actual path for "+ task)
	plt.xlabel("Time")
	plt.ylabel("Error")
	plt.legend()
	plt.show()

if manhattan_arg and filtering_arg:
	compute_manhattan_error(pred_states1, states,"filtering")
if manhattan_arg and smoothing_arg:
	compute_manhattan_error(pred_states2, states,"smoothing")












