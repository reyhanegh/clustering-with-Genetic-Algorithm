# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:33:43 2021

@author: Nava
"""
from sklearn import datasets
import pandas as pd
#from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import pygad
from funcs import plot_data, fitness_func, cluster_data

#load iris data
iris = datasets.load_iris()
data = iris.data[:, 2:]  
target = iris.target

#plot iris data
df = pd.DataFrame(data)
df['target'] = target
plot_data(df, 'iris data')

## normalize data
#scaled_data = data.copy()
#scaler = MinMaxScaler().fit(scaled_data)
#scaled_data = scaler.transform(scaled_data)
##plot normalize data
#scaled_data = pd.DataFrame(scaled_data)
#scaled_data['target'] = target
#plot_data(scaled_data, 'scaled iris data')

num_clusters = 3
feature_vector_length = data.shape[1]
num_genes = num_clusters * feature_vector_length

ga_instance = pygad.GA(num_generations=100,
                       sol_per_pop=10,
                       init_range_low=0,
                       init_range_high=7,
                       num_parents_mating=5,
                       keep_parents=2,
                       num_genes=num_genes,
                       fitness_func=fitness_func,
                       suppress_warnings=True)

ga_instance.run()

best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
print("Best solution is {bs}".format(bs=best_solution))
print("Fitness of the best solution is {bsf}".format(bsf=best_solution_fitness))
print("Best solution found after {gen} generations".format(gen=ga_instance.best_solution_generation))

cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist = cluster_data(best_solution, best_solution_idx)

#scaler2 = MinMaxScaler().fit(cluster_centers)
#cluster_centers = scaler2.transform(cluster_centers)

print("Best solution is {bs}".format(bs=cluster_centers))
plt.figure()
for cluster_idx in range(num_clusters):
    cluster_x = data[clusters[cluster_idx], 0]
    cluster_y = data[clusters[cluster_idx], 1]
    
    plt.scatter(cluster_x, cluster_y)
    plt.scatter(cluster_centers[cluster_idx, 0], cluster_centers[cluster_idx, 1], linewidths=5)
plt.title("Clustering using PyGAD")
plt.show()