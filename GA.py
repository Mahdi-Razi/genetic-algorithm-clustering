import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, euclidean_distances
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Iris dataset
iris = load_iris()
data = iris.data
labels = iris.target
n_data = len(data)
n_clusters = 3

def crossover(parent1, parent2):
    # Generate two random crossover points within the range of the parents lengths
    crossover_point1 = np.random.randint(1, len(parent1) - 1)
    crossover_point2 = np.random.randint(1, len(parent1) - 1)

    # Determine the left and right boundaries for the crossover segment
    l = min(crossover_point1, crossover_point2)
    r = max(crossover_point1, crossover_point2)

    # Create children by combining segments of parent1 and parent2
    child1 = np.concatenate((parent1[:l], parent2[l:r], parent1[r:]))
    child2 = np.concatenate((parent2[:l], parent1[l:r], parent2[r:]))

    return child1, child2


def mutation(chromosome, radius):
    # Create a copy of the original chromosome
    mutated_chromosome = chromosome.copy()
    
    # Choose a random index for mutation
    mutation_i = np.random.randint(n_data - 1)
    
    # Find the data point with the least distance 
    # from mutated point within the 'radius'
    minimum = float('inf')
    index = -1
    l = max(0, mutation_i - radius)
    r = min(n_data - 1, mutation_i + radius)
    for i in range(l, r):
        if i == mutation_i:
            continue
        dist = euclidean_distances(data[mutation_i].reshape(1, -1), data[i].reshape(1, -1))
        if  dist < minimum:
            minimum = dist
            index = i
    
    # Replace the mutated point label with its nearest neighbor in the 'radius'
    mutated_chromosome[mutation_i] = chromosome[index]
    
    return mutated_chromosome


def objective_function(chromosome):
    labels_pred = chromosome.astype(int)
    silhouette_avg = silhouette_score(data, labels_pred)
    return -silhouette_avg  # Negative because we want to minimize (convert silhouette score to a minimization problem)


def heuristic_seed():
    # Initialize an empty list to store centroids
    centroids = []

    # Select a random data point as the first centroid
    centroids.append(data[np.random.randint(data.shape[0]), :])

    seed = np.zeros(n_data)

    # Iterate to find the remaining centroids
    for _ in range(n_clusters - 1):
        dist = []
        # Calculate distances for each data point to it's nearest centroid
        for i in range(data.shape[0]):
            point = data[i]
            d = float('inf')
            for j in range(len(centroids)):
                temp_dist = euclidean_distances(point.reshape(1, -1), centroids[j].reshape(1, -1))
                d = min(d, temp_dist)

            dist.append(d)
        dist = np.array(dist)

        # Select the data point with the maximum distance
        # from current centroids as the next centroid
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)

        # Assign cluster labels based on the nearest centroid for each data point
        for i in range(data.shape[0]):
            point = data[i]
            d = float('inf')

            # Iterate through existing centroids to find the nearest one
            for j in range(len(centroids)):
                temp_dist = euclidean_distances(point.reshape(1, -1), centroids[j].reshape(1, -1))
                if temp_dist < d:
                    d = temp_dist
                    seed[i] = j

    return seed

# Number of generations and population size
num_generations = 200
population_size = 150

# Initialize population
population = np.random.randint(0, n_clusters, size=(population_size, n_data))
population[0] = heuristic_seed()

# Main loop
for generation in range(num_generations):
    # Evaluate fitness (objective) for each chromosome in the population
    fitness = np.array([objective_function(chromosome) for chromosome in population])
    
    # Select the top 50% of the population based on fitness for reproduction
    selected_indices = np.argsort(fitness)[:population_size//2]
    selected_population = population[selected_indices]

    # Crossover: Pairwise crossover among selected chromosomes
    children = []
    for i in range(0, len(selected_population)-1, 2):
        child1, child2 = crossover(selected_population[i], selected_population[i+1])
        children.extend([child1, child2])

    # Mutation: Apply mutation to some of the children
    mutation_rate = 0.5
    num_mutations = int(mutation_rate * len(children))
    mutation_indices = np.random.choice(range(len(children)), size=num_mutations, replace=False)
    for index in mutation_indices:
        children[index] = mutation(children[index], 5)

    # Replace old population with the combined population of selected individuals and children
    population = np.vstack((selected_population, children))

# Select the best chromosome from the final population
best_chromosome = population[np.argmin(fitness)]

# Extract the best labels from the best chromosome
y_pred_GA = best_chromosome.astype(int)

# Evaluate and visualize
sil_score_GA = silhouette_score(data, y_pred_GA)
plt.scatter(data[:, 0], data[:, 1], c=y_pred_GA)
plt.title(f'Genetic Algorithm Clustering\nSilhouette Score: {sil_score_GA:.2f}')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(data)

# Evaluate and visualize
sil_score_kmeans = silhouette_score(data, y_kmeans)
plt.scatter(data[:, 0], data[:, 1], c=y_kmeans)
plt.title(f'K-means Clustering\nSilhouette Score: {sil_score_kmeans:.2f}')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()