import numpy as np
from numpy.linalg import det, inv
from scipy.special import digamma
import pandas as pd
import datetime
import utils
import matplotlib.pyplot as plt

class Bayes_UCB_V:
	def __init__(self, x, t, r):
		self.THRESHOLD = 10e-3
		self.x = x # features of songs - dim : num_of_features x num_of_songs
		self.t = t # last listened times of songs - dim : (num_of_intervals + 2) x num_of_songs
		self.r = r # ratings - dim : 1 x num_of_songs
		self.p,self.N = x.shape
		self.K = t.shape[0]
		# TODO: How to initialize these variables?
		self.lambda_theta_N = np.identity(self.p)
		self.eta_theta_N = np.zeros((self.p,1))
		self.eta_beta_N = np.zeros((self.K,1))
		self.lambda_beta_N = np.identity(self.K)
		self.D_0 = np.identity(self.p)
		self.E_0 = np.identity(self.K)
		self.mu_theta_0 = np.random.random((self.p,1))
		self.mu_beta_0 = np.random.random((self.K,1))
		self.a_0 = 2
		self.b_0 = 2 * 10e-8
		self.a_N = self.a_0
		self.b_N = self.b_0
		self.prev_L = 0.0

		
	def optimize_parameters(self):
		while self.is_converged(self.lower_bound()) == False:
			self.update_theta()
			self.update_beta()
			self.update_tau()
		return self.lambda_theta_N, self.eta_theta_N, self.lambda_beta_N, self.eta_beta_N
	
	def lower_bound(self):       
		return (self.a_0 * np.log(self.b_0) 
			+ (self.a_0 - 1) * (digamma(self.a_N) - np.log(self.b_N))
			- (self.b_0 * self.a_N / self.b_N)
			- (np.log(2 * np.pi)/2)
			- ( np.log(det(self.D_0))/2 )
			+ (self.p / 2) * (digamma(self.a_N) - np.log(self.b_N))
			- (self.a_N / (2 * self.b_N)) * (np.trace(self.D_0.dot(inv(self.lambda_theta_N))) 
											   + (self.mu_theta_0 - self.expected_theta()).T.dot(inv(self.D_0)).dot(self.mu_theta_0 - self.expected_theta())
											 )
			- (self.K * np.log(2 * np.pi) / 2)
			- (np.log(det(self.E_0)) / 2)
			+ (self.K / 2) * (digamma(self.a_N) - np.log(self.b_N))
			- (self.a_N / (2 * self.b_N)) * (np.trace(self.E_0.dot(inv(self.lambda_beta_N))) 
											   + (self.mu_theta_0 - self.expected_theta()).T.dot(inv(self.D_0)).dot(self.mu_theta_0 - self.expected_theta())
											 )
			- (np.log(2 * np.pi)/ 2)
			+ (1 / 2) * (digamma(self.a_N) - np.log(self.b_N))
			- (self.a_N / (2 * self.b_N)) * self.sum1() # Check
			+ (self.a_N / self.b_N) * self.sum2() # Check
			+ (self.K / 2) * (1 + np.log(2 * np.pi))
			+ (1 / 2) * (np.log(det(inv(self.lambda_beta_N))))
			+ (self.p / 2) * (1 + np.log(2 * np.pi))
			+ (1 / 2) * (np.log(det(inv(self.lambda_theta_N))))
			- (self.a_N - 1) * digamma(self.a_N)
			- np.log(self.b_N) + self.a_N        
			)        
					   
	def sum1(self):
		result = self.r.dot(self.r.T)

		for i in range(self.N):
			x_i = self.x[:,i].reshape((self.p,1))
			t_i = self.t[:,i].reshape((self.K,1))
			
			result += x_i.T.dot(self.expected_outer_theta()).dot(x_i).dot(t_i.T).dot(self.expected_outer_beta()).dot(t_i)
			
		return result
	
	def sum2(self):
		result = 0
		
		for i in range(self.N):
			x_i = self.x[:,i].reshape((self.p,1))
			t_i = self.t[:,i].reshape((self.K,1))
			
			r_i = self.r[0][i]
			
			result += r_i*(x_i.T.dot(self.expected_theta()).dot(t_i.T).dot(self.expected_beta())) + self.b_0
			
		return result
					   
	def is_converged(self, L):
		if abs(L - self.prev_L) < self.THRESHOLD:
			return True
		else:
			self.prev_L = L
			return False
	
	def sum_product_along_axis_1(self, First, Second, Beta):
		p,N = First.shape
		K = Second.shape[0]

		result = np.zeros((p,p))
		
		for i in range(N):
			f_i = First[:,i].reshape((p,1))
			s_i = Second[:,i].reshape((K,1))
			
			result += f_i.dot(s_i.T).dot(Beta).dot(s_i).dot(f_i.T)
			
		return result
	
	def sum_product_along_axis_2(self, v_1, First, Second, v_2):
		p, N = First.shape
		K = Second.shape[0]

		result = np.zeros((p,1))
		
		for i in range(N):
			f_i = First[:,i].reshape((p,1))
			s_i = Second[:,i].reshape((K,1))
			r_i = v_1[0][i]
		   
			result += ( r_i * (f_i.dot(s_i.T).dot(v_2)) )
			# print("r", r_i)
			# print(f_i)
			# print(s_i)
			# print(v_2)
			# print(( r_i * (f_i.dot(s_i.T).dot(v_2)) ))
		return result
		
	def update_theta(self):
		exp_tau = self.expected_tau()
		
		self.lambda_theta_N = exp_tau * (inv(self.D_0) + self.sum_product_along_axis_1(self.x, self.t, self.expected_outer_beta()))

		self.eta_theta_N = exp_tau * (inv(self.D_0).dot(self.mu_theta_0)  + self.sum_product_along_axis_2(self.r, self.x, self.t, self.expected_beta()))
		#print("theta", self.eta_theta_N)
	def update_beta(self):               
		exp_tau = self.expected_tau()
		E_0_inv = inv(self.E_0)

		self.lambda_beta_N = exp_tau * (E_0_inv + self.sum_product_along_axis_1(self.t, self.x, self.expected_outer_theta()))              
		
		self.eta_beta_N = exp_tau * ( E_0_inv.dot(self.mu_beta_0) + self.sum_product_along_axis_2(self.r, self.t, self.x, self.expected_theta()))
		#print("beta", self.eta_beta_N)
	def update_tau(self):
		self.a_N = (self.p + self.K + self.N) / 2 + self.a_0
		b_N_update = 0.5*(np.trace(inv(self.D_0).dot(self.expected_outer_theta())) + (self.mu_theta_0.T - 2*self.expected_theta().T).dot(inv(self.D_0)).dot(self.mu_theta_0))

		b_N_update += 0.5*(np.trace(inv(self.E_0).dot(self.expected_outer_beta())) + (self.mu_beta_0.T - 2*self.expected_beta().T).dot(inv(self.E_0)).dot(self.mu_beta_0))
		
		b_N_update += (0.5*self.sum1() - self.sum2() + self.b_0)
		
		self.b_N = b_N_update

	def expected_beta(self):
		return inv(self.lambda_beta_N).dot(self.eta_beta_N)
	
	def expected_theta(self):
		return inv(self.lambda_theta_N).dot(self.eta_theta_N)
	
	def expected_tau(self):
		return self.a_N / self.b_N
	
	def expected_outer_beta(self):
		exp_beta = self.expected_beta()
		return inv(self.lambda_beta_N) + exp_beta.dot(exp_beta.T)
	
	def expected_outer_theta(self):
		exp_theta = self.expected_theta()
		return inv(self.lambda_theta_N) + exp_theta.dot(exp_theta.T)


# Map time 't' to a vector. The epsilon below is the one from the paper
def vectorize(t, epsilon):
#     print(t)
#     print(epsilon.shape)
	v = np.concatenate([t - epsilon, [t,1]])
	v[np.where(v < 0)] = 0
 
	return v

def calculate_quantile(x, t, x_mean_coeff, x_var_coeff, t_mean_coeff, t_var_coeff, alpha, num_samples):
	# sample from theta.T.dot(x) and beta.T.dot(x) and create histogram
	samples_1 = np.random.normal(x.T.dot(x_mean_coeff).item(), x.T.dot(x_var_coeff).dot(x).item(), size = num_samples)
	samples_2 = np.random.normal(t.T.dot(t_mean_coeff).item(), t.T.dot(t_var_coeff).dot(t).item(), size = num_samples)
	
	final_samples = samples_1 * samples_2
	
	# construct histogram
	pdf, intervals = np.histogram(final_samples, 100)
	cdf = np.cumsum(pdf/sum(pdf))
	
	# return quantile 
	for index, value in enumerate(cdf):
		if value >= alpha:
			return intervals[index]


epsilon = np.array([np.power(2,i) for i in np.arange(0, 16, dtype=float)])

class BayesUCB:
	
	def __init__(self, simulation=True): 
		self.epsilon = epsilon
		self.recommended_song_ids = []
		self.simulation = simulation
		self.util = utils.Util()
		self.recommend_song()

	def recommend(self):
		return self.recommended_song_ids[-1]
	
	def recommend_song(self):
		if len(self.recommended_song_ids) == 0:
			song_id = np.random.randint(self.util.get_number_of_songs())
		else:
			song_id = self.recommended_song_candidate
		self.recommended_song_ids.append(song_id)
		print(song_id)
		self.util.add_recommendation(song_id, self.simulation)

		
	def feedback(self, rating):
		self.util.add_rating(rating)
		t = self.util.get_history_times()
		x = self.util.get_features_of_history()
		y = self.util.get_ratings()
		self.recommended_song_candidate = self.driver(x, t, y, epsilon)
		self.recommend_song()
		
		
	def driver(self, features, last_listened, ratings, epsilon):
		time_vectors = np.zeros((len(last_listened),len(epsilon) + 2))
		for i, x in enumerate(last_listened):
			time_vectors[i,:] = vectorize(x, epsilon)

		time_vectors = time_vectors.T

		ratings = ratings.reshape((1,len(ratings)))

		bayesUCB_V = Bayes_UCB_V(features, time_vectors, ratings)

		lambda_theta_N, eta_theta_N, lambda_beta_N, eta_beta_N = bayesUCB_V.optimize_parameters()

		# section 4.2.3 : page 11
		x_mean_coeff = inv(lambda_theta_N).dot(eta_theta_N)
		x_var_coeff = inv(lambda_theta_N)
		#print(lambda_theta_N)
		#print(eta_theta_N)


		t_mean_coeff = inv(lambda_beta_N).dot(eta_beta_N)
		t_var_coeff = inv(lambda_beta_N)

		# get quantiles for each song and choose argmax
		N = features.shape[1]
		quantiles = []
		for i in range(N):
			x_i = features[:,i].reshape((features.shape[0],1))
			t_i = time_vectors[:,i].reshape((time_vectors.shape[0],1))

			quantiles.append(calculate_quantile(x_i, t_i, x_mean_coeff, x_var_coeff, t_mean_coeff, t_var_coeff, 1 - 1/(N + 1), 10000))

		return np.argmax(quantiles)
