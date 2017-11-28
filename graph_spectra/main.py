import numpy as np
import pandas as pd
from random import randint
import random
import argparse
import time
import itertools, collections
from numpy.linalg import inv, eig
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="synthetic")
parser.add_argument('--k', type=int, default=4)
parser.add_argument('--toy', type=bool, default=False)

args = parser.parse_args()

class Preprocessing(object):
	"""docstring for preprocessing"""
	def __init__(self):
		super(preprocessing, self).__init__()
	
	@staticmethod
	def importing_the_graph(dataset):
		results = []
		if dataset == "real":
			with open('file1.txt') as inputfile:
			    for line in inputfile:
			        results.append(line.strip().split(','))
		else:
			with open('file2.txt') as inputfile:
			    for line in inputfile:
			    	results.append(line.strip().split(','))
		return np.array(results[0:50])
	
	@staticmethod
	def visualizing_graph(graph):
		G = nx.from_edgelist(graph)
		plt.figure(1)
		nx.draw_networkx(G)
		plt.show()
		return

	@staticmethod
	def visualizing_clustered_graph(graph, clusters, k):
		G = nx.from_edgelist(graph)
		plt.figure(1)
		values = []
		for node in G.nodes():
			values.append(clusters.get(int(node), k+1))
		nx.draw(G, cmap=plt.get_cmap('jet'), node_color=values)
		plt.show()
		return

class SpectralClustering(object):
	"""docstring for SpectralClustering"""
	def __init__(self, nodes, graph, k):
		super(SpectralClustering, self).__init__()
		self.nodes = nodes
		self.graph = graph
		self.k = k
		
	def neighbourhood(self, n):
			neigh = []
			for edge in self.graph:
				if edge[0] == n:
					neigh.append(edge[1])
				if edge[1] == n:
					neigh.append(edge[0])
			neigh = list(set(neigh))
			return neigh

	def bfs_shortest_path(self, n1, n2):
		explored = []
		queue = [[n1]]
		while queue:
			path = queue.pop(0)
			node = path[-1]
			if node not in explored:
				neighbours = self.neighbourhood(node)
				for neighbour in neighbours:
					new_path = list(path)
					new_path.append(neighbour)
					queue.append(new_path)
					if neighbour == n2:
						return len(new_path) - 1
				explored.append(node)
		return len(self.nodes)

	def similarity_matrix(self):
		A = np.zeros((len(self.nodes), len(self.nodes)))
		for i in range(0, len(self.nodes)):
			for j in range(i+1, len(self.nodes)):
				A[i][j] = self.bfs_shortest_path(self.nodes[i], self.nodes[j])
		return A + A.T

	def diagonal_matrix(self, A):
		D = np.zeros((len(self.nodes), len(self.nodes)))
		for i in range(0, len(A)):
			D[i][i] = np.sum(A[i,:])
		return D

	def compute_eigenvectors(self, L):
		V = eig(L)
		return V[1][:, 0:self.k]

	def clustering(self, Y):
		kmeans = KMeans(n_clusters=self.k)
		kmeans.fit(Y)
		labels = kmeans.predict(Y)
		return np.insert(Y, Y.shape[1], labels, axis=1)
		
	def run(self):
		A = self.similarity_matrix()
		D = self.diagonal_matrix(A)
		L = inv(np.sqrt(D)).dot(A).dot(inv(np.sqrt(D)))
		X = self.compute_eigenvectors(L)
		Y = np.zeros(X.shape)
		for i in range(0, X.shape[0]):
			for j in range(0, X.shape[1]):
				Y[i][j] = X[i][j] / np.sqrt(np.sum(np.power(X[i,:],2)))
		Y_clustered = self.clustering(Y)
		clusters = np.zeros((len(self.nodes), 2))
		for i, node in enumerate(self.nodes):
			clusters[i][0] = node
			clusters[i][1] = Y_clustered[i][self.k]
		return clusters

def main():

	""" Data preprocessing """
	graph = np.array(Preprocessing.importing_the_graph(args.dataset))

	# dropping the weights
	if args.dataset == "synthetic":
		graph = np.delete(graph, 2, 1)
	if args.toy == True:
		graph = np.array([[1,2], [1, 3], [3,2], [4,5], [6,7], [8, 9], [9,10], [10,8]])

	# visualizing the graph
	Preprocessing.visualizing_graph(list(graph))

	# all unique nodes
	nodes = np.unique(np.union1d(graph[:,0], graph[:,1]))

	# running the algorithm
	spectral = SpectralClustering(nodes=nodes, graph=graph, k=args.k)
	clusters = spectral.run()

	# creating the dictionary for visualization
	print(clusters)
	clusters_dict = {}
	for i, node in enumerate(clusters):
		clusters_dict[int(node[0])] = clusters[i][1]
	print(clusters_dict)

	Preprocessing.visualizing_clustered_graph(list(graph), clusters_dict, args.k)

	return
main()