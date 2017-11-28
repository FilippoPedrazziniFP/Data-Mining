# MinHashing : A class MinHashing that builds a minHash signature (in the form of 
# a vector or a set) of a given length n from a given set of integers (a set of hashed shingles).

import random

class MinHashing(object):
  def __init__():
    super(MinHashing, self).__init__()

  def next_prime(n):
    a = n
    b = 2*n

    for p in range(a, b):
      for i in range(2, p):
        if p % i == 0:
          break
        else:
          return p
    return None

  def random_coeff(k, max_value):
    rand_list = []
    while k > 0:
      rand_ix = random.randint(1, max_value) 
      while rand_ix in rand_list:
        rand_ix = random.randint(1, max_value) 
      rand_list.append(rand_ix)
      k = k - 1  
    return rand_list







