import numpy as np
import utils


class LinUCB:
    def __init__(self, alpha, datapath):
        self.util = utils.Util(datapath)
        self.song_features = self.util.get_features_and_times()
        self.d = self.song_features.shape[0]
        self.K = self.song_features.shape[1]  # length of feature vector
        self.alpha = alpha
        self.A = np.zeros((self.d, self.K, self.K))
        for i in range(self.d):
            self.A[i] = np.identity(self.K)
        self.b = np.zeros((self.d, self.K))
        self.theta_hat = np.zeros(self.K)
        self.choosen_song_index = 0  # random initial value
        self.p = np.zeros(self.d)
        self.norms = []
        self.ratings = []
        self.choices = []
        self.rewards = []
        self.epsilon = 0.2

    def recommend(self):
        for a in range(self.d):
            x = self.song_features[a]
            A_inv = np.linalg.inv(self.A[a])
            self.theta_hat = A_inv.dot(self.b[a])
            ta = x.T.dot(A_inv).dot(x)
            a_upper_ci = self.alpha * np.sqrt(ta)
            a_mean = self.theta_hat.dot(x)
            a_mean_with_sum = sum([self.theta_hat[i] * x[i] for i in range(self.theta_hat.shape[0])])
            self.p[a] = a_mean_with_sum + a_upper_ci

        self.p = self.p + (np.random.random(len(self.p)) * 0.00001)
        recommended_song = self.p.argmax()
        self.choices.append(recommended_song)
        self.choosen_song_index = recommended_song
        A_inv = np.linalg.inv(self.A[recommended_song])
        theta_hat = A_inv.dot(self.b[recommended_song])
        a_mean = self.theta_hat.dot(self.song_features[recommended_song])
        self.util.add_expected_rating(a_mean)
        self.util.add_recommendation(recommended_song)
        return recommended_song

    def feedback(self, rating):
        self.util.add_rating(rating)
        self.song_features = self.util.get_features_and_times()
        x = self.song_features[self.choosen_song_index]

        self.ratings.append(rating)
        self.A[self.choosen_song_index] += np.outer(x, x)
        self.b[self.choosen_song_index] += rating * x
