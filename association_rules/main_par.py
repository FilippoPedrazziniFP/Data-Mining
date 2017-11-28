import numpy as np
import pandas as pd
import argparse
import time
import itertools, collections

# multi processes version
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--baskets', type=int, default=1000)
parser.add_argument('--s_thr', type=float, default=0.01)
parser.add_argument('--c_thr', type=float, default=0.5)
parser.add_argument('--i_thr', type=float, default=0.5) 
parser.add_argument('--proc', type=int, default=4)
parser.add_argument('--k_step', type=int, default=5)
parser.add_argument('--toy', type=bool, default=False)
args = parser.parse_args()

class Util(object):
	"""docstring for ClassName"""
	def __init__(self, arg):
		super(Util, self).__init__()
		self.arg = arg

	@staticmethod
	def importing_the_dataset(number_of_baskets):
		results = []
		with open('file.txt') as inputfile:
		    for line in inputfile:
		        results.append(line.strip().split(' '))
		return results[0:number_of_baskets]

	@staticmethod
	def display_association_rule(interest, rule):
		print("Rule: [basket containing : %s --> %s ] with interest: %s" % (rule[0],rule[1], interest))
		return

class APriori(object):
	"""docstring for APriori"""
	def __init__(self, item_dict, support_threshold, all_unique_items, dataset, processes, thr_c, thr_i):
		super(APriori, self).__init__()
		self.item_dict = item_dict
		self.support_threshold = support_threshold
		self.all_unique_items = all_unique_items
		self.dataset = dataset
		self.thr_c = thr_c
		self.thr_i = thr_i

	def support(self, basket):
		value = 0
		for transaction in self.dataset:
			items = set(transaction)
			basket = set(basket)
			if basket.issubset(items):
			 	value = value + 1 
		return value

	def first_filter(self, item):
		if self.item_dict[item] > self.support_threshold:
			return item

	def ith_filter(self, element):
		sup = self.support(element)
		if sup > self.support_threshold:
			return element

	def ith_generation(self, items, items_signleton, length):
		items_gen = []
		for item in items:
			max_item = max(item)
			for element in items_signleton:
				if element > max_item:
					items_tmp = list(item)
					items_tmp.append(element)
					items_tmp_pairs = list(itertools.combinations(items_tmp, length))
					if all((el in items) for el in items_tmp_pairs):
						items_gen.append(tuple(items_tmp))
		return items_gen

	def compute_association_rules(self, candidates, length):
		rules = self.generate_rules(candidates, length)
		for rule in rules:
			confidence = self.confidence(rule)
			if confidence > self.thr_c:
				interest = self.interest(rule[1], confidence)
				if interest > self.thr_i:
					Util.display_association_rule(interest, rule)
		return

	def generate_rules(self, candidates, length):
		rules = []
		for candidate in candidates:
			all_rules = list(itertools.combinations(candidate, length-1))
			for i, rule in enumerate(all_rules):
				rule_tmp = []
				for element in candidate:
					if element not in rule:
						rule_tmp.append(rule)
						rule_tmp.append(element)
				rules.append(rule_tmp)
		return rules

	def confidence(self, rule):
		sup = self.support(rule[0])
		union_rule = list(rule[0])
		union_rule.append(rule[1])
		sup_union = self.support(union_rule)
		return sup_union/sup

	def interest(self, element, confidence):
		return confidence - (self.item_dict[element]/len(self.dataset))

def main():

	if args.toy == True:
		dataset = [[1,2,3,4],[1,2,3],[1,2,3,4],[1,2],[1],[1],[1,3,4]]
	else:
		dataset = Util.importing_the_dataset(args.baskets)
	
	print("Number of baskets: ", len(dataset))

	""" Preprocessing """

	# flatting the nparray
	all_items = np.hstack(np.array(dataset))
	print("Number of items in the dataset: ", all_items.shape)

	# considering just unique items
	all_unique_items = np.unique(all_items)
	print("Number of unique items in the dataset: ", all_unique_items.shape)

	# building the dictionary with the number of occurrences
	item_dict = {}
	for item in all_unique_items:
		count = 0
		for basket in dataset:
			if item in basket:
				count = count + 1
		item_dict[item] = count

	# defining the support
	if args.toy == True:
		support_threshold = 1
		args.i_thr = 0.3
	else:
		support_threshold = args.s_thr*all_unique_items.shape[0]
	print("Support threshold: ", support_threshold)

	""" A PRIORI METHOD (3 steps) with the computation of the association rules"""

	a_priori = APriori(item_dict=item_dict, 
		support_threshold=support_threshold, 
		all_unique_items=all_unique_items,
		dataset=dataset,
		processes=args.proc,
		thr_c=args.c_thr,
		thr_i=args.i_thr)

	start_time = time.time()
	
	items_filtered = []
	items_gen = []

	# singleton
	items_signleton = []
	
	i = 1
	while (i < args.k_step):

		if i == 1:
			# First filter
			with Pool(processes=args.proc) as pool:
				items_signleton = pool.map(a_priori.first_filter, all_unique_items, int(len(all_unique_items)/args.proc))
			items_signleton = list(set(items_signleton))
			items_signleton = list(filter(None, items_signleton))
			print("Elements after the %s filter: %s" %(i, len(items_signleton)))

			# generating candidate pairs
			items_gen = list(itertools.combinations(items_signleton, i+1))
			print("Elements after the %s generation: %s" %(i,len(items_gen)))
			
			i = i + 1

		else:

			""" the filters are done in parallel using multiprocessing pyhton library """

			# Ith filter
			with Pool(processes=args.proc) as pool:
				items_filtered = pool.map(a_priori.ith_filter, items_gen, int(len(items_gen)/args.proc))
			items_filtered = list(set(items_filtered))
			items_filtered = list(filter(None, items_filtered))
			print("Elements after the %s filter: %s" %(i, len(items_filtered)))

			""" after generating all the itemsets of length i, we compute the association rules """

			# compute association rules
			a_priori.compute_association_rules(items_filtered, i)

			# Ith generation
			items_gen = a_priori.ith_generation(items_filtered, items_signleton, i)
			print("Elements after the %s generation: %s" %(i, len(items_gen)))
			
			i = i + 1

main()