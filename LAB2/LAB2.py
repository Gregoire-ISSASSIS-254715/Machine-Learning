##########################
########## LAB2 ##########
##########################

##########################
# Import required libraries
import copy
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans

##########################
# Exercise 1 - K-means from scratch

# Variables declarations
k = 4
n_of_points = 60
n_of_iterations = 200


# Functions definitions
def split_points(points: np.array, n_of_point_groups: int) -> np.array:
    changed_points = copy.copy(points)
    index = np.arange(len(points))
    groups_index = np.split(index, n_of_point_groups)

    for id_group, group_index in enumerate(groups_index):
        changed_points[group_index] = points[group_index] + 5 * id_group

    return changed_points


# Variables declaration
points = np.random.rand(n_of_points, 2) * 5
points = split_points(points, 3)


def initialize_clusters(points: np.array, k_clusters: int) -> np.array:
    vector_with_all_indexes = np.arange(points.shape[0])
    vector_with_all_indexes = np.random.permutation(vector_with_all_indexes)
    required_indexes = vector_with_all_indexes[:k_clusters]
    return points[required_indexes]


def calculate_metric(points: np.array, centroid: np.array) -> np.array:
    return np.square(norm(points - centroid, axis=1))


def compute_distances(points: np.array, centroids_points: np.array) -> np.array:
    return np.asarray([calculate_metric(points, centroid) for centroid in centroids_points])


def assign_centroids(distances: np.array):
    return np.argmin(distances, axis=1)


def calculate_objective(cluster_belongs: np.array, distances: np.array) -> np.array:
    distances = distances.T
    selected_min = distances[np.arange(len(distances)), cluster_belongs]
    return np.sum(selected_min)


def calculate_new_centroids(points: np.array, clusters_belongs: np.array, n_of_clusters: int) -> np.array:
    new_clusters = []
    for cluster_id in range(n_of_clusters):
        j = np.where(clusters_belongs == cluster_id)
        points_sel = points[j]
        new_clusters.append(np.mean(points_sel, axis=0))

    return np.array(new_clusters)


def fit(points: np.array, n_of_centroids: int, n_of_oterations: int, error: float = 0.001) -> tuple:
    centroid_points = initialize_clusters(points, n_of_centroids)
    last_objective = 10000

    for n in range(n_of_oterations):
        distances = compute_distances(points, centroid_points)
        cluster_belongs = np.argmin(distances, axis=0)

        objective = calculate_objective(cluster_belongs, distances)

        if abs(last_objective - objective) < error:
            break

        last_objective = objective

        centroid_points = calculate_new_centroids(points, cluster_belongs, n_of_centroids)

    return centroid_points, last_objective


centroids, _ = fit(points, 3, n_of_iterations)

# Create the figure
plt.figure()
plt.scatter(points[:, 0], points[:, 1])
plt.scatter(centroids[:, 0].T, centroids[:, 1].T)

# Display the figure
plt.show()
##########################


##########################
# Exercise 2 - Implementing an Elbow Method

# Variables declarations
k_all = range(2, 10)    # Create a vector with values contained between 2 and 10
all_objective = []      # Create an empty list

# Objective function definition
for n_of_cluster in k_all:
    _, objective = fit(points, n_of_cluster, n_of_iterations)
    all_objective.append(objective)

# Create the figure
plt.figure()
plt.plot(k_all, all_objective)

# Rename the axis
plt.xlabel('K clusters')
plt.ylabel('Sum of squared distance')

# Display the figure
plt.show()
##########################


##########################
# Exercise 3 - Image Compression

# Load the image
loaded_image = imread('fish.jpg')

# Show the image
plt.imshow(loaded_image)
plt.show()

# Function definition
def compress_image(image: np.array, number_of_colours: int) -> np.array:
    # Get the matrix dimensions (height, width, 3 RGB sub-pixel values)
    original_shape = image.shape

    # Change the number of dimensions (from 3 to 2)
    image_reshaped = np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))

    # Compute kmeans with the wanted number of colours
    kmeans = KMeans(n_clusters=number_of_colours, random_state=0).fit(image_reshaped)

    # Predict for each sub-pixel value the nearer cluster
    image_new = kmeans.predict(image_reshaped)
    print("\nSub-pixel nearer cluster value :", image_new)

    # Display labels in the figure
    cluster_labels = kmeans.labels_
    colors = kmeans.cluster_centers_.astype('uint8')

    # Create new image with clustered values
    new_image = colors[cluster_labels].reshape(original_shape)

    return new_image


# Compress the image
compressed_image = compress_image(loaded_image, 5)

# Show the compressed image
plt.figure()
plt.imshow(compressed_image)
plt.show()
##########################
