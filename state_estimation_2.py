import random
import numpy as np
import math
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


simulation_arg=int(sys.argv[1])
estimation_arg=int(sys.argv[2])
control=sys.argv[3]
sensing=sys.argv[4]
euclidean_arg=int(sys.argv[5])
ellipse_arg=int(sys.argv[6])
velocities_arg=int(sys.argv[7])
double_tracking=int(sys.argv[8])


def uncertainty_ellipse(mean,cov,ax,facecolor=None,**kwargs):
	pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
	ellipse = Ellipse((0, 0), width=np.sqrt(1 + pearson) * 2, height=np.sqrt(1 - pearson) * 2,facecolor=facecolor,**kwargs)
	transf = transforms.Affine2D() \
		.rotate_deg(45) \
		.scale(np.sqrt(cov[0, 0]) , np.sqrt(cov[1, 1])) \
		.translate(mean[0], mean[1])

	ellipse.set_transform(transf + ax.transData)
	return ax.add_patch(ellipse)


class Motion_Model():
	def __init__(self, x_init, y_init, Vx_init, Vy_init):

		self.R=np.eye(4)
		self.R[2,2],self.R[3,3]=0.0001,0.0001

		self.A=np.eye(4)
		self.A[0,2],self.A[1,3]=1,1

		self.init_state=np.array([x_init,y_init,Vx_init,Vy_init]).reshape(4,1)

	def measurement_prob(self,belief,control,t):
		if control=="sine":
			self.A[0,2]=abs(math.sin(20*t))
			self.A[1,3]=abs(math.cos(20*t))
		mean=belief.mean
		sigma=belief.sigma
		mean= np.dot(self.A,mean).reshape(4,1)
		sigma=np.matmul(np.matmul(self.A,sigma),self.A.transpose()) + self.R
		return mean, sigma

class Sensor_Model():
	def __init__(self):

		self.Q= 100*np.eye(2)

		self.C=np.array([[1,0,0,0],[0,1,0,0]])

	def sensation_prob(self, belief,z):
		mean= belief.mean
		sigma=belief.sigma

		K=np.linalg.inv(np.matmul(np.matmul(self.C,sigma),self.C.reshape(4,2)) + self.Q)
		
		K=np.matmul(np.matmul(sigma,self.C.reshape(4,2)),K)

		mean=mean + np.matmul(K,z-np.matmul(self.C,mean))

		sigma= np.matmul(np.eye(4)- np.matmul(K,self.C), sigma)

		return mean, sigma


def simulate1(motion,sensor,time_steps):
	states,observations=[],[]

	A=motion.A
	R=motion.R
	state=motion.init_state
	Q=sensor.Q
	C=sensor.C

	states.append(state.flatten())
	sensor_error=np.array([np.random.normal(0,Q[0,0]**(0.5)),np.random.normal(0,Q[1,1]**(0.5))])
	obs=np.dot(C,state).flatten() + sensor_error 
	observations.append(obs)

	for t in range(time_steps):
		motion_error=np.array([np.random.normal(0,R[0,0]**(0.5)),np.random.normal(0,R[1,1]**(0.5)),abs(np.random.normal(0,R[2,2]**(0.5))),abs(np.random.normal(0,R[3,3]**(0.5)))])
		state=np.dot(A,state.reshape(4,1)).flatten() + motion_error
		# state=np.dot(A,state.reshape(4,1)).flatten()
		states.append(state)
		sensor_error=np.array([np.random.normal(0,Q[0,0]**(0.5)),np.random.normal(0,Q[1,1]**(0.5))])
		# print(sensor_error)
		obs=np.dot(C,state.reshape(4,1)).flatten() + sensor_error
		# obs=np.dot(C,state.reshape(4,1)) 
		observations.append(obs)

	return states,observations

motion=Motion_Model(0,0,0,0)
sensor=Sensor_Model()
time_steps=200

states,observations=simulate1(motion,sensor,time_steps)

if simulation_arg:
	plt.plot([x[0] for x in observations],[y[1] for y in observations],color="yellow",linestyle='dashed',label="observed")
	plt.plot([x[0] for x in states],[y[1] for y in states],color="blue",linestyle='dashed',label="actual")
	plt.title("Simulation- Observed and Actual trajectories")
	plt.xlabel("x-coordinate")
	plt.ylabel("y-coordinate")
	plt.legend()
	plt.show()

class Belief():
	def __init__(self, x, y, Vx, Vy, s_x, s_y, s_vx, s_vy):
		self.mean=np.array([x,y,Vx,Vy]).reshape(4,1)
		self.sigma=np.eye(4)
		self.sigma[0,0], self.sigma[1,1], self.sigma[2,2], self.sigma[3,3]=s_x, s_y, s_vx, s_vy



class BayesFilter():
	def __init__(self,motion,sensor):
		self.motion=motion
		self.sensor=sensor
		self.init=motion.init_state.flatten()
		self.init_belief=Belief(self.init[0], self.init[1], self.init[2], self.init[3], 1,1,0.0001,0.0001)

	def update_from_sensor(self,z):

		mean,sigma=self.sensor.sensation_prob(self.init_belief,z)
		self.init_belief.mean=mean
		self.init_belief.sigma=sigma

		return self.init_belief.mean.flatten(),sigma

		

	def update_from_measure(self, control,t):
		mean,sigma=self.motion.measurement_prob(self.init_belief,control,t)
		self.init_belief.mean=mean
		self.init_belief.sigma=sigma
		
		return self.init_belief.mean.flatten(),sigma

	def step(self,control,t,z):
		sense,sigma=None,None

		if sensing=="broken":
			if t<10 or (t>=20 and t<30) or t>=40:
				sense,sigma1=self.update_from_sensor(z)
		else:
			sense,sigma1=self.update_from_sensor(z)

		measure,sigma2=self.update_from_measure(control,t)

		if sensing=="broken":
			sigma=sigma2
		else:
			sigma=sigma1

		return sense,measure,sigma

class object():
	def __init__(self,motion,sensor):
		self.motion=motion
		self.sensor=sensor

def simulate2(motion, sensor, time_steps):
	actual,estimated,observed=[], [], []
	bayes=BayesFilter(motion,sensor)
	covs=[]

	for t in range(time_steps):
		sensor_error=np.array([np.random.normal(0,sensor.Q[0,0]**(0.5)),np.random.normal(0,sensor.Q[1,1]**(0.5))]).reshape(2,1)
		z=np.matmul(sensor.C,bayes.init_belief.mean) + sensor_error

		motion_error=np.array([np.random.normal(0,motion.R[0,0]**(0.5)),np.random.normal(0,motion.R[1,1]**(0.5)),abs(np.random.normal(0,motion.R[2,2]**(0.5))),abs(np.random.normal(0,motion.R[3,3]**(0.5)))])
		state=np.matmul(motion.A,bayes.init_belief.mean) + motion_error.reshape(4,1)

		sense,measure,sigma= bayes.step(control,t,z)
		actual.append(state.flatten())
		if sensing=="broken":
			estimated.append(measure)
		else:
			estimated.append(sense)
		observed.append(z)
		covs.append(sigma)


	return actual,estimated,observed,covs

motion=Motion_Model(10,10,1,1)
sensor=Sensor_Model()
time_steps=200

actual,estimated,observed,covs=simulate2(motion,sensor,time_steps)

if estimation_arg:

	plt.plot([x[0] for x in observed],[y[1] for y in observed],color="yellow" ,label="observed")
	plt.plot([x[0] for x in actual],[y[1] for y in actual],color="green", label="actual")
	plt.plot([x[0] for x in estimated],[y[1] for y in estimated],color="blue", label="estimated")

	plt.title("Trajectories after filtering")
	plt.xlabel("x-coordinate")
	plt.ylabel("y-coordinate")
	plt.legend()
	plt.show()

def plot_ellipse(mean,cov,title):
	fig, ax_nstd = plt.subplots(figsize=(6, 6))
	ax_nstd.axvline(c='grey', lw=1)
	ax_nstd.axhline(c='grey', lw=1)
	for i in range(len(mean)):
		uncertainty_ellipse(mean[i],cov[i], ax_nstd,facecolor="white" , edgecolor='firebrick')
	plt.title(title)
	plt.plot([x[0] for x in mean],[x[1] for x in mean])
	plt.show()

if ellipse_arg:
	plot_ellipse(estimated,covs,"Uncertainty Ellipse for discontinuous sensor measurements")

if velocities_arg:
	plt.plot([x[2] for x in actual],[y[3] for y in actual],color="green", label="actual")
	plt.plot([x[2] for x in estimated],[y[3] for y in estimated],color="blue", label="estimated")

	plt.title("Velocities-actual and estimated")
	plt.xlabel("x-coordinate")
	plt.ylabel("y-coordinate")
	plt.legend()
	plt.show()


def euclidean(x,y):
	return np.sqrt(np.sum(np.square(x-y)))

if euclidean_arg:

	plt.plot(np.linspace(0,time_steps,time_steps),[euclidean(x[0:2],y[0:2]) for x,y in zip(actual,estimated)],marker='*')
	plt.xlabel("Time_step")
	plt.ylabel("Error")
	plt.title("Euclidean distance error for sinusoidal control policy")
	plt.show()


def mahalanobis(val,mean,sigma):
	y=val-mean
	d=np.matmul(np.matmul(y.flatten(),sigma.transpose()),y)
	return d

def simulate_double_tracking(object1, object2, time_steps):
	actual1,estimated1,observed1=[], [], []
	actual2,estimated2,observed2=[], [], []
	bayes1=BayesFilter(object1.motion, object1.sensor)
	bayes2=BayesFilter(object2.motion, object2.sensor)

	for t in range(time_steps):
		se1=np.array([np.random.normal(0,object1.sensor.Q[0,0]**(0.5)),np.random.normal(0,object1.sensor.Q[1,1]**(0.5))]).reshape(2,1)
		se2=np.array([np.random.normal(0,object2.sensor.Q[0,0]**(0.5)),np.random.normal(0,object2.sensor.Q[1,1]**(0.5))]).reshape(2,1)
		obs1=np.matmul(object1.sensor.C,bayes1.init_belief.mean) + se1
		obs2=np.matmul(object2.sensor.C,bayes2.init_belief.mean) + se2

		me1=np.array([np.random.normal(0,object1.motion.R[0,0]**(0.5)),np.random.normal(0,object1.motion.R[1,1]**(0.5)),np.random.normal(0,object1.motion.R[2,2]**(0.5)),np.random.normal(0,object1.motion.R[3,3]**(0.5))])
		s1=np.matmul(object1.motion.A,bayes1.init_belief.mean) + me1.reshape(4,1)


		me2=np.array([np.random.normal(0,object2.motion.R[0,0]**(0.5)),np.random.normal(0,object2.motion.R[1,1]**(0.5)),np.random.normal(0,object2.motion.R[2,2]**(0.5)),np.random.normal(0,object2.motion.R[3,3]**(0.5))])
		s2=np.matmul(object2.motion.A,bayes2.init_belief.mean) + me2.reshape(4,1)

		# print(s2.flatten())

		z1= np.argsort([mahalanobis(obs1,bayes1.init_belief.mean[0:2,:],bayes1.init_belief.sigma[0:2,0:2])[0],mahalanobis(obs2,bayes1.init_belief.mean[0:2,:],bayes1.init_belief.sigma[0:2,0:2])[0]])[0]
		if z1==0:
			z1=obs1
		else:
			z1=obs2
		z2= np.argsort([mahalanobis(obs1,bayes2.init_belief.mean[0:2,:],bayes2.init_belief.sigma[0:2,0:2])[0],mahalanobis(obs2,bayes2.init_belief.mean[0:2,:],bayes2.init_belief.sigma[0:2,0:2])[0]])[0]
		if z2==0:
			z2=obs1
		else:
			z2=obs2

		a1,e1,_=bayes1.step(control,t,z1)
		a2,e2,_=bayes2.step(control,t,z2)

		actual1.append(s1.flatten())
		actual2.append(s2.flatten())
		estimated1.append(e1)
		estimated2.append(e2)
		observed1.append(z1)
		observed2.append(z2)

	return actual1,estimated1,observed1, actual2,estimated2,observed2

motion1=Motion_Model(10,10,1,1)
sensor1=Sensor_Model()
object1=object(motion1,sensor1)

motion2=Motion_Model(20,25,1,1)
sensor2=Sensor_Model()

motion2.R[2,2],motion2.R[3,3]=0.01,0.01
sensor2.Q=sensor2.Q*10
object2=object(motion2,sensor2)

a1,e1,z1,a2,e2,z2=simulate_double_tracking(object1,object2,200)

if double_tracking:
	plt.plot([x[0] for x in e1],[x[1] for x in e1],linestyle='dashed',label="estimated_1")
	plt.plot([x[0] for x in e2],[x[1] for x in e2],linestyle='dashed',label="estimated_2")
	plt.plot([x[0] for x in a1],[x[1] for x in a1],linestyle='dashed',label="actual_1")
	plt.plot([x[0] for x in a2],[x[1] for x in a2],linestyle='dashed',label="actual_2")
	plt.title("Tracking 2 objects using a Data association strategy")
	plt.legend()
	plt.show()





















	



