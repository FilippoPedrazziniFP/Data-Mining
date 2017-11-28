#A class LSH that implements the LSH technique: given a collection of minhash signatures (integer vectors) and 
#a similarity threshold t, the LSH class (using banding and hashing) finds all candidate pairs of signatures 
#that agree on at least fraction t of their components.
from compare_sign import CompareSign
import random

class LSH(object):

	def __init__(self, hash_functions, c, threshold, shg, documents_number, signatures_matrix, text):
		super(LSH, self).__init__()
		self.hash_functions = hash_functions
		self.c = c
		self.threshold = threshold
		self.shg = shg
		self.documents_number = documents_number
		self.signatures_matrix = signatures_matrix
		self.text = text

	def compute_lsh(self):

		print("LSH phase: ")
		bands = 5
		rows = int(self.hash_functions/bands)

		# defining the hash function
		a = random.randint(1, self.c)
		b = random.randint(1, self.c)

		similar_items_idx = []
		for i in range(0, bands):
			# one array of buckets for each band
			buckets = []
			for j in range(0, self.documents_number):
				value = 0
				for r in range(i, i + rows):
					value = value + (self.signatures_matrix[r][j]*a + b)
				hashed_value = value % self.c
				buckets.append(hashed_value)
			for b in range(0, len(buckets)):
				for z in range(b+1, len(buckets)):
					if buckets[b] == buckets[z]:
						# checking if similarity has already been computed
						if (b not in similar_items_idx) & (z not in similar_items_idx):
							sim = CompareSign.similarity(self.signatures_matrix[:,b], self.signatures_matrix[:,z], self.hash_functions)
							if sim > self.threshold:
								print("text1: ", self.text[b][0:100])
								print("text2: ", self.text[z][0:100])
								print("common elements: ", sim)
								print("jaggard similarity: ", CompareSign.jaggard_similarity(self.text[b], self.text[z], self.shg))
								print()
						similar_items_idx.append(b)
						similar_items_idx.append(z)

		return