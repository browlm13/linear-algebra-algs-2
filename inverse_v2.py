#!/usr/bin/env python3

"""

	Compute the inverse of A, X=A^-1, 
		by LU factorization 
		and then solving a column at a time for AX = I

	TODO: Error for singular matrix

"""

__filename__ = "inverse.py"
__author__ = "L.J. Brown"

# internal
import logging

# external
import numpy as np
import scipy.linalg

# initilize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inverse(A, tol=10**(-14), inplace=False):
	""" 
		Compute the inverse of the matrix A and store in place if specified. 

		:param A: nxn non-singular matrix to compute the inverse of.
		:param tol: Defualt set to 10^(-14), consider  values bellow this threshold to be zero.
		:param inplace: Defualt set to False. Store Ai (A inverse) in parameter A when True, create new matrix when set to False, return Ai in either case.
		:returns: Ai (A inverse).
	"""
	n = A.shape[0]

	# LU decomposition -- raise ValueError for singular matrix A
	try:
		LU, piv = scipy.linalg.lu_factor(A)

		# enforce magnitude of diagonal values are above given tolernce (round off)
		for di in np.diag(A):
			if abs(di) <= tol: raise ValueError

	except ValueError:
		logger.error("Error 'Singular Matrix' passed method: %s" % inverse.__name__)

	# initalize Ai depending on paramter 'inplace'
	if inplace:
		Ai = np.empty()
	else:
		Ai = A 				# Ai is refrence to A

	# initilize column vector of identity matrix to zeros 
	ei = np.zeros(shape=(n,))

	# solve for A^-1 and store in A
	for i in range(n):
		ei[i] = 1
		Ai[:,i] = scipy.linalg.lu_solve((LU, piv), ei)
		ei[i] = 0

	# return Ai (A inverse)
	return Ai

if __name__ == "__main__":

	# create random matrix A
	n = 3
	MAX_VAL = 2
	A = np.random.rand(n,n)*np.random.randint(low=-MAX_VAL, high=MAX_VAL)

	print(A)
	inverse(A)

	print(A)


