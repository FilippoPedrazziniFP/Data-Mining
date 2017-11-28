# Shingling : A class Shingling that constructs 
# k–shingles of a given length k (e.g., 10) from a given document, 
# computes a hash value for each unique shingle, and represents 
# the document in the form of an ordered set of its hashed k-shingles.

# example
# The set of all contiguous sequences of k tokens
# "a rose is a rose is a rose"
# { (a,rose,is,a), (rose,is,a,rose), (is,a,rose,is), (a,rose,is,a), (rose,is,a,rose) } = { (a,rose,is,a), (rose,is,a,rose), (is,a,rose,is) }

import itertools
import hashlib
import binascii

class Shingling(object):

	def shingle(text, k, char=False):
		
		if char == True:
			# split the string into separate words
			tokenized_text = list(text)
			# article id to drop -- considering the first dataset with id in the beginning
			tokenized_text = tokenized_text[6:]
		else:
			# split the string into separate words
			tokenized_text = text.split()
			# article id to drop -- considering the first dataset with id in the beginning
			del(tokenized_text[0])
		shingle_text = []
		for i in range(len(tokenized_text) - k + 1):
			shingle_text.append(tokenized_text[i:i+k])
		# remove duplicates
		shingle_text.sort()
		un_shg_txt = list(shingle_text for shingle_text,_ in itertools.groupby(shingle_text))
		# from a list to a string
		sum_strings = []
		for i, text in enumerate(un_shg_txt):
			sum_string = text[0]
			for i in range(1, len(text)):
				sum_string = sum_string + " " + text[i]
			sum_strings.append(sum_string)
		return sum_strings



