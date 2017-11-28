# CompareSignature : A class CompareSignatures that estimates similarity of two integer vectors – 
# nhash signatures – as a fraction of components, in which they agree.
from shingling import Shingling

class CompareSign(object):
	"""docstring for CompareSign"""
	def __init__(self, arg):
		super(CompareSign, self).__init__()
		self.arg = arg

	def similarity(sig1, sig2, hashes):
		count = 0
		for k in range(0, hashes):
			if sig1[k] == sig2[k]:
				count = count + 1
		return count/hashes

	def jaggard_similarity(txt1, txt2, k):
		txt1 = set(Shingling.shingle(txt1, k))
		txt2 = set(Shingling.shingle(txt2, k))
		return (len(txt1.intersection(txt2)) / len(txt1.union(txt2)))
