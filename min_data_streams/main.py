import numpy as np
import pandas as pd
from random import randint
import random
import argparse
import time
import itertools, collections
import networkx as nx
import matplotlib.pyplot as plt

# Patent Cit. dataset with 7515023 Triangles and Fraction of closed triangles 0.02343 -- 16518948 edges
# Adolescent health dataset with 4694 Triangles -- 12,969 edges

parser = argparse.ArgumentParser()
parser.add_argument('--memory', type=int, default=250)
parser.add_argument('--edges', type=int, default=300)
parser.add_argument('--toy', type=bool, default=False)
parser.add_argument('--dataset', type=str, default="health")

args = parser.parse_args()

class Preprocessing(object):
	"""docstring for Preprocessing"""
	def __init__(self, arg):
		super(Preprocessing, self).__init__()
		self.arg = arg
	
	@staticmethod
	def importing_the_graph(number_of_edges, dataset):
		results = []
		if dataset == "health":
			with open('file.txt') as inputfile:
			    for line in inputfile:
			        results.append(line.strip().split(' '))
			# dropping the first 2 rows (description)
			del(results[0:2])
		else:
			with open('patents.txt') as inputfile:
			    for line in inputfile:
			    	results.append(line.strip().split('\t'))
			# dropping the first 2 rows (description)
			del(results[0:4])
			print(results[0:10])
		return results[0:number_of_edges]
	
	@staticmethod
	def visualizing_graph(graph):
		G = nx.from_edgelist(graph)
		print("Networkx result: ", float(sum(list(nx.triangles(G).values())))/3)
		plt.figure(1)
		nx.draw_networkx(G)
		plt.show()
		return

class Triest(object):
	"""docstring for ClassName"""
	def __init__(self, graph, vertices):
		super(Triest, self).__init__()
		self.graph = graph
		self.S = []
		self.vertices = vertices
		self.global_counter = 0
		self.local_counters = {self.vertices[i]: 0 for i in range(0, self.vertices.shape[0])}

	def neighbourhood(self, u):
		neigh = []
		for edge in self.S:
			if edge[0] == u:
				neigh.append(edge[1])
			if edge[1] == u:
				neigh.append(edge[0])
		neigh = list(set(neigh))
		return neigh

	def add_edge_to_s(self, edge):
		self.S.append(tuple(edge))
		return

	def substitute_edge(self, j, edge):
		self.S[j] = tuple(edge)
		return
		
	def update_counters(self, edge, sign):
		neigh_u = self.neighbourhood(edge[0])
		neigh_v = self.neighbourhood(edge[1])

		# considering just the intersection
		neigh = list(set(neigh_u) & set(neigh_v))

		for vtx in neigh:
			if sign == True:
				self.global_counter = self.global_counter + 1
				self.local_counters[vtx] = self.local_counters[vtx] + 1
				self.local_counters[edge[0]] = self.local_counters[edge[0]] + 1
				self.local_counters[edge[1]] = self.local_counters[edge[0]] + 1
			else:
				self.global_counter = self.global_counter - 1
				self.local_counters[vtx] = self.local_counters[vtx] - 1
				self.local_counters[edge[0]] = self.local_counters[edge[0]] - 1
				self.local_counters[edge[1]] = self.local_counters[edge[0]] - 1
		return

	def update_counters_improved(self, edge, t, M):
		neigh_u = self.neighbourhood(edge[0])
		neigh_v = self.neighbourhood(edge[1])

		# considering just the intersection
		neigh = list(set(neigh_u) & set(neigh_v))
		incr = max(1, (t-1)*(t-2)/(M*(M-1)))

		for vtx in neigh:
			self.global_counter = self.global_counter + incr
			self.local_counters[vtx] = self.local_counters[vtx] + incr
			self.local_counters[edge[0]] = self.local_counters[edge[0]] + incr
			self.local_counters[edge[1]] = self.local_counters[edge[0]] + incr
		return

def main():

	if args.toy == True:
		graph = np.array([[1,2], [1, 3], [2,3], [2,4], [2,5], [4,5], [6, 4], [4, 7]])

	else:
		# transforming it into a nparray
		graph = np.array(Preprocessing.importing_the_graph(args.edges, args.dataset))
		
		print(graph.shape)
		# dropping the weights
		if args.dataset == "health":
			graph = np.delete(graph, 2, 1)
		print(graph.shape)

	Preprocessing.visualizing_graph(list(graph))

	# flatting the nodes (all the unique nodes)
	vertices = np.unique(np.union1d(graph[:,0], graph[:,1]))

	""" TRIEST BASE """

	triest = Triest(graph=graph, vertices=vertices)

	start_time = time.time()
	e_factor = []
	global_counters = []
	t = 0
	for i, edge in enumerate(graph): 
		e_factor.append(max(1, t*(t-1)*(t-2)/(args.memory*(args.memory-1)*(args.memory-2))))
		global_counters.append(triest.global_counter)
		t = t + 1
		if t <= args.memory:
			triest.add_edge_to_s(edge)
			triest.update_counters(edge=edge, sign=True)
		else:
			if random.random() > (args.memory/t):
				j = randint(0, args.memory-1)
				triest.substitute_edge(j, edge)
				triest.update_counters(edge=edge, sign=False)

	if t > args.memory:
		counters = np.asarray(global_counters)
		e_factor = np.asarray(e_factor)
		final_value = np.sum(np.multiply(counters, e_factor))/t
	else:
		final_value = triest.global_counter*e_factor[-1]
	print("Number of Triangles BASE: ", final_value)
	print("Processing time: %s seconds" % (time.time() - start_time))

	""" TRIEST IMPRoved """

	triest = Triest(graph=graph, vertices=vertices)

	start_time = time.time()
	
	t = 0
	for i, edge in enumerate(graph):
		t = t + 1
		triest.update_counters_improved(edge=edge, t=t, M=args.memory)
		if t <= args.memory:
			triest.add_edge_to_s(edge)	
		else:
			if random.random() > (args.memory/t):
				j = randint(0, args.memory-1)
				triest.substitute_edge(j, edge)

	print("Number of Triangles IMPR: ", triest.global_counter)
	print("Processing time: %s seconds" % (time.time() - start_time))


main()