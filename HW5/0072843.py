import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[-6.0, -1.0],
                        [-3.0, +2.0],
                        [+3.0, +2.0],
                        [+6.0, -1.0]])

group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +4.0]],
                              [[+2.4, -2.0],
                               [-2.0, +2.4]],
                              [[+2.4, +2.0],
                               [+2.0, +2.4]],
                              [[+0.4, +0.0],
                               [+0.0, +4.0]]])

# read data into memory
data_set = np.genfromtxt("hw05_data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 4

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below
    initial_means = np.genfromtxt("hw05_initial_centroids.csv", delimiter=",")
    assert initial_means.shape == (K, X.shape[1])

    # Select the first K rows as initial means
    means = initial_means[:K]

    # Assign points by distance to initial means
    distances = dt.cdist(X, means)
    assignments = np.argmin(distances, axis=1)

    # Calculate covariances for each cluster
    covariances = np.array([np.cov(X[assignments == k].T) for k in range(K)])

    # Calculate priors for each cluster
    priors = np.array([np.mean(assignments == k) for k in range(K)])
    # your implementation ends above
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(data_points, num_clusters, cluster_means, cluster_covariances, cluster_priors):

    # Initialize cluster assignments
    data_point_assignments = np.zeros(data_points.shape[0])
    iteration=100
    for _ in range(iteration):

        cluster_responsibilities = np.array([
            cluster_priors[cluster_index] * stats.multivariate_normal(
                cluster_means[cluster_index], cluster_covariances[cluster_index]
            ).pdf(data_points)
            for cluster_index in range(num_clusters)
        ]).T 

        sum_responsibilities = np.sum(cluster_responsibilities, axis=1).reshape(-1, 1)
        cluster_responsibilities /= sum_responsibilities

        for cluster_index in range(num_clusters):
            responsibility_for_cluster = cluster_responsibilities[:, cluster_index].reshape(-1, 1)
            total_responsibility = np.sum(responsibility_for_cluster)

            cluster_means[cluster_index] = np.sum(responsibility_for_cluster * data_points, axis=0) / total_responsibility
            deviation_from_mean = data_points - cluster_means[cluster_index]
            cluster_covariances[cluster_index] = (deviation_from_mean.T @ (responsibility_for_cluster * deviation_from_mean)) / total_responsibility
            cluster_priors[cluster_index] = total_responsibility / data_points.shape[0]

        data_point_assignments = np.argmax(cluster_responsibilities, axis=1)

    return cluster_means, cluster_covariances, cluster_priors, data_point_assignments


means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    plt.figure(figsize=(8, 8))
    colors = ['red', 'blue', 'green', 'purple']

    [plt.scatter(X[assignments == i, 0], X[assignments == i, 1], s=10, color=colors[i], label=f'Cluster {i+1}') for i in range(K)]


    x = np.linspace(-8, 8, 151)
    y = np.linspace(-8, 8, 151)

    X_mesh, Y_mesh = np.meshgrid(x, y)
    pos = np.empty(X_mesh.shape + (2,))
    pos[:, :, 0] = X_mesh; pos[:, :, 1] = Y_mesh

    [plt.contour(X_mesh, Y_mesh, stats.multivariate_normal.pdf(pos, mean=means[i], cov=covariances[i]), levels=[0.01], colors=colors[i]) for i in range(K)]
    [plt.contour(X_mesh, Y_mesh, stats.multivariate_normal.pdf(pos, mean=group_means[i], cov=group_covariances[i]), levels=[0.01], colors='k', linestyles='dashed') for i in range(K)]

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.show()
    plt.savefig("0072843.pdf")

# Example usage:
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)