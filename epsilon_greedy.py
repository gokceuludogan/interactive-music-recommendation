import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import utils


class EpsilonGreedy:
    def __init__(self, epsilon, simulation=True):
        self.util = utils.Util()
        self.epsilon = epsilon
        self.recommended_song_ids = []
        self.simulation = simulation
        self.cumulative_regret = 0
        self.recommend_song()
        self.recommended_song_candidate = 0

    def recommend(self):
        return self.recommended_song_ids[-1]

    def recommend_song(self):
        if len(self.recommended_song_ids) == 0 or self.epsilon > np.random.rand():
            song_id = np.random.randint(self.util.get_number_of_songs())  # random choice
        else:
            song_id = self.recommended_song_candidate  # greedy choice
        self.recommended_song_ids.append(song_id)
        self.util.add_recommendation(song_id, self.simulation)

    def feedback(self, rating):
        self.util.add_rating(rating)
        t = self.util.get_all_times()
        x = self.util.get_all_features()
        theta, s = self.calculate_theta_s()
        self.recommended_song_candidate = np.argmax(theta.T.dot(x) * (1 - np.exp(-t / s)))
        self.calculate_cumulative_regret(theta, s)
        self.recommend_song()

    def calculate_cumulative_regret(self, theta, s):
        y = self.util.get_ratings()
        t = self.util.get_history_times()
        x = self.util.get_features_of_history()
        y_model = theta.T.dot(x) * (1 - np.exp(-t / s))
        self.util.add_expected_rating(y_model[-1])
        self.cumulative_regret = np.average(y - y_model)

    def calculate_theta_s(self):
        initial_values = np.zeros(self.util.get_number_of_features() + 1)
        initial_values[-1] = 1
        x = self.util.get_features_of_history()
        y = self.util.get_ratings()
        t = self.util.get_history_times()
        position, _, _ = fmin_l_bfgs_b(self.optimization, x0=initial_values, args=(x, y, t), approx_grad=True)
        theta = position[:-1]
        s = position[-1]
        return theta, s

    def optimization(self, params, *args):
        x = args[0]
        y = args[1]
        t = args[2]
        theta = params[:-1]
        s = params[-1]
        y_model = theta.T.dot(x) * (1 - np.exp(-t / s))
        error = y - y_model
        return sum(error ** 2)
