import numpy as np
import utils
import epsilon_greedy
import linucb
from bayesucb import BayesUCB
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def calculate_rating(theta, s, song_features, song_times):
    # print('theta', theta)
    # print('song_features', song_features)
    # print('song_times', song_times)
    # print('s', s)
    # song_times /= 100
    return theta.T.dot(song_features) * (1 - np.exp(-song_times / s)) * 10


def generate_simulation(song_names, method, theta, s, length):
    model = BayesUCB()
    if method == 'Random':
        model = epsilon_greedy.EpsilonGreedy(1.0)
    elif method == 'Greedy':
        model = epsilon_greedy.EpsilonGreedy(0.2)
    elif method == 'BayesUCB':
        model = BayesUCB()
    elif method == 'LinUCB':
        alpha = 0.1
        model = linucb.LinUCB(alpha)
    for i in range(length):
        recommended_song = model.recommend()
        print("Recommended song: ", i, recommended_song, ' '.join(song_names.iloc[recommended_song]))
        features, times = model.util.get_features_and_times_of_song(recommended_song)
        # print(times)
        rating = calculate_rating(theta, s, features, times)
        print('rating: ', rating)
        model.feedback(rating)

    cum_regret = model.util.get_cumulative_regret()
    fig1 = plt.figure(1, figsize=(10, 5))
    plt.plot(cum_regret)
    plt.title("Cumulative Regret")
    fig1.savefig(method + "_regret.png")
    plt.close(fig1)

    cumulative_average_rating = model.util.get_cumulative_average_rating()
    fig2 = plt.figure(1, figsize=(10, 5))
    plt.plot(cumulative_average_rating)
    plt.title("Cumulative Average Rating")
    fig2.savefig(method + "_rating.png")
    plt.close(fig2)


def main():
    np.random.seed(12)
    data, song_names = utils.get_data("data/last_fm_songs_with_features.csv")
    data, song_names = shuffle(data, song_names, random_state=0)
    theta = np.random.random(data.shape[1])
    # s = 100 fine: 10 exploitation
    s = 1000
    length = 200
    # methods = ["Random", "Greedy", "LinUCB", "BayesUCB"]
    methods = ["BayesUCB"]
    for method in methods:
        generate_simulation(song_names, method, theta, s, length)


if __name__ == '__main__':
    main()
