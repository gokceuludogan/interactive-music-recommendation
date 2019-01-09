import numpy as np
import utils
from models import linucb, bayesucb, epsilon_greedy
import matplotlib.pyplot as plt
from os.path import join

def calculate_rating(theta, s, song_features, song_times):
    return theta.T.dot(song_features) * (1 - np.exp(-song_times / s)) * 10

def generate_simulation(song_names, method, theta, s, length, datapath):
    if method == 'Random':
        model = epsilon_greedy.EpsilonGreedy(1.0, datapath)
    elif method == 'Greedy':
        model = epsilon_greedy.EpsilonGreedy(0.2, datapath)
    elif method == 'LinUCB':
        alpha = 0.1
        model = linucb.LinUCB(alpha, datapath)
    else:
        model = bayesucb.BayesUCB(datapath)

    for i in range(length):
        recommended_song = model.recommend()
        print("Recommended song: ", i, recommended_song, ' '.join(song_names.iloc[recommended_song]))
        features, times = model.util.get_features_and_times_of_song(recommended_song)
        rating = calculate_rating(theta, s, features, times)
        print('Rating: ', rating)
        model.feedback(rating)

    cum_regret = model.util.get_cumulative_regret()
    fig1 = plt.figure(1, figsize=(10, 5))
    plt.plot(cum_regret)
    plt.title("Cumulative Regret")
    fig1.savefig(join("output", method + "_regret.png"))
    plt.close(fig1)

    cumulative_average_rating = model.util.get_cumulative_average_rating()
    fig2 = plt.figure(1, figsize=(10, 5))
    plt.plot(cumulative_average_rating)
    plt.title("Cumulative Average Rating")
    fig2.savefig(join("output", method + "_rating.png") )
    plt.close(fig2)


def main():
    filepath = "data/last_fm_songs_with_features.csv"
    np.random.seed(12)
    data, song_names = utils.get_data(filepath)
    theta = np.random.random(data.shape[1])
    s = 1000
    length = 30
    methods = ["Random", "Greedy", "LinUCB", "BayesUCB"]
    for method in methods:
        print("Starting simulation for model", method)
        print("======================================")
        generate_simulation(song_names, method, theta, s, length, filepath)
        print("End of  simulation for model", method)
        print("======================================")

if __name__ == '__main__':
    main()
