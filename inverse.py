#!/usr/bin/env python3

"""

	Compute the inverse of A, X=A^-1, 
		by LU factorization 
		and then solving a column at a time for AX = I

"""

# external
import numpy as np
import scipy.linalg

def column_oriented_backward_substitution(U,y):

	for j in range(U.shape[1]-1,-1,-1):
		y[j] = y[j]/U[j,j]
		for i in range(j):
			y[i] -= U[i,j]*y[j]
		
		
	return y


def unit_column_oriented_forward_substitution(L,b):

	for j in range(L.shape[1]): 
		for i in range(j+1,L.shape[0]):
			b[i] -= L[i,j]*b[j]

	return b


def inverse(A):

	n = A.shape[0]

	# LU decomposition
	LU, piv = scipy.linalg.lu_factor(A)

	A_inv = np.empty(shape=(n,n))
	ei = np.zeros(shape=(n,))
	for i in range(n):
		ei[i] = 1
		A_inv[:,i] = scipy.linalg.lu_solve((LU, piv), ei)
		ei[i] = 0

	I_prime = A_inv @ A

	# average error
	E = abs(np.eye(n)-I_prime)
	error = E.sum(axis=0).sum()
	print(error)

	return A_inv



if __name__ == "__main__":

	# create random matrix A
	n = 100
	MAX_VAL = 10
	A = np.random.rand(n,n)*np.random.randint(low=-MAX_VAL, high=MAX_VAL)

	inverse(A)
