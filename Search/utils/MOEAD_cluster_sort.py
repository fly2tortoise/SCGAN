import numpy as np
import sys
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
def Prefer_cluster_MOEAD(args, random_F,  N, weight):
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

    # Top (N - cluster_select) by Tchebycheff distance
    moead_selected = sorted_indices_tchebycheff[:(N - args.cluster_select)]

    # Middle slice [N - cluster_select, N + cluster_select) for hierarchical clustering
    remaining_indices = sorted_indices_tchebycheff[(N - args.cluster_select):(N +  args.cluster_select)]
    # print(len(remaining_indices))
    def hierarchical_clustering_selection(data, num_clusters):
        # print(num_clusters)
        cosine_distance = pdist(data, metric='cosine')
        Z = linkage(cosine_distance, 'ward')
        clusters = fcluster(Z, t=num_clusters, criterion='maxclust')
        selected_indices = []

        for cluster in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster)[0]
            selected_indices.append(cluster_indices[0])

        return np.array(selected_indices)
    cluster_selected =  hierarchical_clustering_selection(random_F[remaining_indices],
                                                              num_clusters= args.cluster_select)
    if len(cluster_selected) !=  args.cluster_select:
        print("Cluster count mismatch")
        sys.exit()
    final_selected_indices = np.concatenate((moead_selected, remaining_indices[cluster_selected]))

    return final_selected_indices

