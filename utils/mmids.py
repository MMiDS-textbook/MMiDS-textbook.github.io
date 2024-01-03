# Description: Utility functions for MMIDS


# Libraries

import numpy as np
from numpy import linalg as LA
from numpy.random import default_rng
import matplotlib.pyplot as plt
rng = default_rng(535)


# k-means clustering

def opt_reps(X, k, assign):
    """
    Calculate the representative point for each cluster.

    Parameters:
    - X (numpy.ndarray): The input data matrix of shape (n, d).
    - k (int): The number of clusters.
    assign (numpy.ndarray): The assignment array of shape (n,) where assign[i] represents the cluster assignment of data point X[i].

    Returns:
    numpy.ndarray: The representative points for each cluster, of shape (k, d).
    """
    (n, d) = X.shape
    reps = np.zeros((k, d))
    for i in range(k):
        in_i = [j for j in range(n) if assign[j] == i]             
        reps[i,:] = np.sum(X[in_i,:],axis=0) / len(in_i)
    return reps

import numpy as np
from numpy import linalg as LA

def opt_clust(X, k, reps):
    """
    Assign the given data points to the optimal cluster.

    Parameters:
    - X: numpy array, shape (n, d), representing the data points
    - k: int, the number of clusters
    - reps: numpy array, shape (k, d), representing the initial cluster centroids

    Returns:
    - assign: numpy array, shape (n,), representing the cluster assignments for each data point
    """

    (n, d) = X.shape
    dist = np.zeros(n)
    assign = np.zeros(n, dtype=int)
    for j in range(n):
        dist_to_i = np.array([LA.norm(X[j,:] - reps[i,:]) for i in range(k)])
        assign[j] = np.argmin(dist_to_i)
        dist[j] = dist_to_i[assign[j]]
    G = np.sum(dist ** 2)
    print(G) # Print current objective to monitor progress
    return assign

def kmeans(X, k, maxiter=10):
    """
    Perform k-means clustering on the given data.

    Parameters:
    - X: numpy.ndarray
        The input data array of shape (n, d), where n is the number of data points and d is the number of dimensions.
    - k: int
        The number of clusters to create.
    - maxiter: int, optional
        The maximum number of iterations to perform. Default is 10.

    Returns:
    - assign: numpy.ndarray
        The cluster assignments for each data point, represented as an array of shape (n,).

    """
    (n, d) = X.shape
    assign = rng.integers(0,k,n)
    reps = np.zeros((k, d), dtype=int)
    for iter in range(maxiter):
        # Step 1: Optimal representatives for fixed clusters
        reps = opt_reps(X, k, assign) 
        # Step 2: Optimal clusters for fixed representatives
        assign = opt_clust(X, k, reps) 
    return assign


# k-NN regression

def knnregression(x,y,k,xnew):
    n = len(x)
    closest = np.argsort([np.absolute(x[i]-xnew) for i in range(n)])
    return np.mean(y[closest[0:k]])


# Algorithms for linear systems

def backsubs(R,b):
    m = b.shape[0]
    x = np.zeros(m)
    for i in reversed(range(m)):
        x[i] = (b[i] - np.dot(R[i,i+1:m],x[i+1:m]))/R[i,i]
    return x

def forwardsubs(L,b):
    m = b.shape[0]
    x = np.zeros(m)
    for i in range(m):
        x[i] = (b[i] - np.dot(L[i,0:i],x[0:i]))/L[i,i]
    return x

def cholesky(B):
    n = B.shape[0] 
    L = np.zeros((n, n))
    for j in range(n):
        L[j,0:j] = forwardsubs(L[0:j,0:j],B[j,0:j])
        L[j,j] = np.sqrt(B[j,j] - LA.norm(L[j,0:j])**2)
    return L 

def ls_by_chol(A, b):
    L = cholesky(A.T @ A)
    z = forwardsubs(L, A.T @ b)
    return backsubs(L.T, z)

def gramschmidt(A):
    (n,m) = A.shape
    Q = np.zeros((n,m))
    R = np.zeros((m,m))
    for j in range(m):
        v = np.copy(A[:,j])
        for i in range(j):
            R[i,j] = np.dot(Q[:,i], A[:,j])
            v -= R[i,j]*Q[:,i]
        R[j,j] = LA.norm(v)
        Q[:,j] = v/R[j,j]
    return Q, R

def householder(A, b):
    n, m = A.shape
    R = np.copy(A)
    Qtb = np.copy(b)
    for k in range(m):
    
        # computing z
        y = R[k:n,k]
        e1 = np.zeros(n-k)
        e1[0] = 1
        z = np.sign(y[0]) * LA.norm(y) * e1 + y
        z = z / LA.norm(z)
        
        # updating R
        R[k:n,k:m] = R[k:n,k:m] - 2 * np.outer(z, z) @ R[k:n,k:m]
        
        # updating Qtb
        Qtb[k:n] = Qtb[k:n] - 2 * np.outer(z, z) @ Qtb[k:n]
    
    return R[0:m,0:m], Qtb[0:m]

