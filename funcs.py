# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 02:27:57 2021

@author: Nava
"""
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy

num_clusters = 3
feature_vector_length = 2
num_genes = num_clusters * feature_vector_length
iris = datasets.load_iris()
data = iris.data[:, 2:] 
def plot_data(data,title):
    df1 = data[data.target == 0]
    df2 = data[data.target == 1]
    df3 = data[data.target == 2]
    plt.figure()
    plt.scatter(df1[0], df1[1], color = 'green', label='cls1')
    plt.scatter(df2[0], df2[1], color = 'red', label='cls2')
    plt.scatter(df3[0], df3[1], color = 'blue', label='cls3')
    plt.title(title)
    

def euclidean_distance(X, Y):
    return numpy.sqrt(numpy.sum(numpy.power(X - Y, 2), axis=1))

def cluster_data(solution, solution_idx):
    global num_clusters, feature_vector_length, data
    cluster_centers = []
    all_clusters_dists = []
    clusters = []
    clusters_sum_dist = []

    for clust_idx in range(num_clusters):
        cluster_centers.append(solution[feature_vector_length*clust_idx:feature_vector_length*(clust_idx+1)])
        cluster_center_dists = euclidean_distance(data, cluster_centers[clust_idx])
        all_clusters_dists.append(numpy.array(cluster_center_dists))

    cluster_centers = numpy.array(cluster_centers)
    all_clusters_dists = numpy.array(all_clusters_dists)

    cluster_indices = numpy.argmin(all_clusters_dists, axis=0)
    for clust_idx in range(num_clusters):
        clusters.append(numpy.where(cluster_indices == clust_idx)[0])
        if len(clusters[clust_idx]) == 0:
            clusters_sum_dist.append(0)
        else:
            clusters_sum_dist.append(numpy.sum(all_clusters_dists[clust_idx, clusters[clust_idx]]))

    clusters_sum_dist = numpy.array(clusters_sum_dist)

    return cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist

def fitness_func(solution, solution_idx):
    _, _, _, _, clusters_sum_dist = cluster_data(solution, solution_idx)

    fitness = 1.0 / (numpy.sum(clusters_sum_dist) + 0.00000001)

    return fitness
