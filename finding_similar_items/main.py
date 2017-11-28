import numpy as np
import pandas as pd
import argparse
from shingling import Shingling
from min_hashing import MinHashing
from compare_sign import CompareSign
from lsh import LSH
import time

# dataset : https://github.com/chrisjmccormick/MinHash/blob/master/data
parser = argparse.ArgumentParser()
parser.add_argument('--na', type=int, default=100)
parser.add_argument('--hashes', type=int, default=10)
parser.add_argument('--thr', type=int, default=0.5)
parser.add_argument('--shg_char', type=bool, default=False)
parser.add_argument('--shg', type=int, default=4)
parser.add_argument('--lsh', type=bool, default=True)
args = parser.parse_args()

def importing_the_dataset(number_of_articles):
	text = []
	dataFile = "./data/articles_" + str(number_of_articles) + ".train"
	f = open(dataFile, "rU")
	for i in range(0, number_of_articles): 
		words = f.readline()
		text.append(words)
	f.close()
	return text

def main():

	print("Importing the Dataset")
	# importing the dataset
	text = importing_the_dataset(args.na)
	documents_number = len(text)

	print("Shingling phase")
	# shingling all the documents
	texts_shingled = []
	for element in text:
		text_shingled = Shingling.shingle(text=element, k=args.shg, char=args.shg_char)
		texts_shingled.append(text_shingled)

	# flatting the list of shingles
	flat_texts_shingled = np.hstack(np.array(texts_shingled))
	print("number of shingles: ", flat_texts_shingled.shape[0])
	unique_flat_texts_shingled = np.unique(flat_texts_shingled)
	max_value = unique_flat_texts_shingled.shape[0]
	print("number of unique shingles: ", max_value)

	# building the dictionary with { unique shingle : number }
	shingle_dic = {unique_flat_texts_shingled[i]: i for i in range(0, unique_flat_texts_shingled.shape[0])}

	# shingle into integers
	text_num = texts_shingled
	for i, txt in enumerate(texts_shingled):
		for j, shg in enumerate(txt):
			text_num[i][j] = shingle_dic[shg]

	print("MinHashing phase")
	# number of hash functions
	hash_functions = args.hashes

	# For each of the hash functions, generate a different coefficient 'a' and 'b'.   
	a = MinHashing.random_coeff(hash_functions, max_value)
	b = MinHashing.random_coeff(hash_functions, max_value)
	# next prime of greatest value
	c = MinHashing.next_prime(max_value)

	# min hashing - building the signatures matrix with [hashing_functions, number of documents]
	signatures_matrix = np.zeros((hash_functions, documents_number))
	for i, txt in enumerate(text_num):
		for j in range(0, hash_functions):
			min_hash = c + 1
			for element in txt:
				hashed_value = (a[j] * element + b[j]) % c 
				if hashed_value < min_hash:
				    min_hash = hashed_value
			signatures_matrix[j][i] = min_hash
	print("min hashing matrix shape: ", signatures_matrix.shape)

	print("Similarity Matrix phase")
	# compuitng the similarity matrix (common signatures)
	similarities = np.zeros((documents_number, documents_number))
	print(similarities.shape)
	for i in range(0, documents_number):
		for j in range(i+1, documents_number):
			similarities[i][j] = CompareSign.similarity(signatures_matrix[:,i], signatures_matrix[:,j], hash_functions)
	
	# mirroring the matrix (all similarity matrixes are symmetric)
	similarities = similarities + similarities.T

	# picking just the most similar items based on a threshold
	threshold = args.thr

	# printing the most similar items
	for i in range(0, documents_number):
		for j in range(i+1, documents_number):
			sim_tmp = similarities[i][j]
			if sim_tmp > threshold:
				print("text1: ", text[i][0:100])
				print("text2: ", text[j][0:100])
				print("common elements: ", sim_tmp)
				print("jaggard similarity: ", CompareSign.jaggard_similarity(text[i], text[j], args.shg))
				print()

	# lsh approximation
	if args.lsh == True:
		lsh = LSH(hash_functions=hash_functions, 
					c=c, 
					threshold=threshold,
					shg=args.shg,
					documents_number=documents_number, 
					signatures_matrix=signatures_matrix, 
					text=text)
		lsh.compute_lsh()
main()

