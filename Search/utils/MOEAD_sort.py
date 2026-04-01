import numpy as np

def Prefer_MOEAD(random_F,  N, weight):
    def normalize_data(F):
        F_min = np.min(F, axis=0)
        F_max = np.max(F, axis=0)
        return (F - F_min) / (F_max - F_min)
    random_F = normalize_data(random_F)
    # weights = np.array([1/3, 1/3, 1/3])
    # weights = np.array([3/10, 3/10, 3/10])
    weights = weight
    ideal_point = np.min(random_F, axis=0)
    def calculate_tchebycheff_distance(F, weights, ideal_point):
        tchebycheff_distances = np.max(weights * np.abs(F - ideal_point), axis=1)
        return tchebycheff_distances
    tchebycheff_distances = calculate_tchebycheff_distance(random_F, weights, ideal_point)
    sorted_indices_tchebycheff = np.argsort(tchebycheff_distances)

    return sorted_indices_tchebycheff[:N]

