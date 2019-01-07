import numpy as np
import utils
import epsilon_greedy
import linucb
from bayesucb import BayesUCB

def calculate_rating(theta, s, song_features, song_times):
	# print('theta', theta)
	# print('song_features', song_features)
	# print('song_times', song_times)
	# print('s', s)
	#song_times /= 100 
	return theta.T.dot(song_features) * (1 - np.exp(-song_times/s)) * 10

def generate_simulation(method, theta, s, length):
	if method == 'Random':
		model = epsilon_greedy.EpsilonGreedy(1.0)
	elif method == 'Greedy':
		model = epsilon_greedy.EpsilonGreedy(0.5)
	elif method == 'BayesUCB':
		model = BayesUCB()
	elif method == 'LinUCB':
		alpha = 0.001
		model = linucb.LinUCB(alpha)
	for i in range(length):
		recommended_song = model.recommend()
		print("Recommended song: ", recommended_song, ' '.join(song_names.iloc[recommended_song]))
		features, times = model.util.get_features_and_times_of_song(recommended_song)
		#print(times)
		rating = calculate_rating(theta, s, features, times)
		print('rating: ', rating)
		model.feedback(rating)


data, song_names = utils.get_data("data/last_fm_songs_with_features_shuf.csv")
theta = np.random.random(data.shape[1]) 
# s = 10 # bayes exploitation
s = 100
length = 100
#methods = ["Random", "Greedy", "LinUCB", "BayesUCB"]
methods = ["BayesUCB"]
for method in methods:
	generate_simulation(method, theta, s, length)