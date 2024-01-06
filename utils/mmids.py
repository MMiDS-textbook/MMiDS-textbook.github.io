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

def knnregression(x, y, k, xnew):
    """
    Perform k-nearest neighbors regression.

    Parameters:
    - x (array-like): The input feature values.
    - y (array-like): The target values.
    - k (int): The number of nearest neighbors to consider.
    - xnew (float): The new input feature value for prediction.

    Returns:
    - float: The predicted target value based on k-nearest neighbors regression.
    """
    n = len(x)
    closest = np.argsort([np.absolute(x[i] - xnew) for i in range(n)])
    return np.mean(y[closest[0:k]])


# Algorithms for linear systems

def backsubs(R, b):
    """
    Perform back substitution to solve the system of linear equations Rx = b.

    Parameters:
    - R (numpy.ndarray): Upper triangular matrix representing the coefficients of the linear equations.
    - b (numpy.ndarray): Column vector representing the constants of the linear equations.

    Returns:
    - x (numpy.ndarray): Column vector representing the solution to the system of linear equations.
    """
    m = b.shape[0]
    x = np.zeros(m)
    for i in reversed(range(m)):
        x[i] = (b[i] - np.dot(R[i, i + 1:m], x[i + 1:m])) / R[i, i]
    return x


def forwardsubs(L, b):
    """
    Solve a lower triangular linear system using forward substitution.

    Parameters:
    L (numpy.ndarray): The lower triangular matrix of shape (m, m).
    b (numpy.ndarray): The right-hand side vector of shape (m,).

    Returns:
    x (numpy.ndarray): The solution vector of shape (m,).
    """
    m = b.shape[0]
    x = np.zeros(m)
    for i in range(m):
        x[i] = (b[i] - np.dot(L[i, 0:i], x[0:i])) / L[i, i]
    return x


def cholesky(B):
    """
    Perform Cholesky decomposition on a given matrix.

    Parameters:
    B (numpy.ndarray): The input matrix.

    Returns:
    numpy.ndarray: The lower triangular matrix L such that B = LL^T.
    """
    n = B.shape[0] 
    L = np.zeros((n, n))
    for j in range(n):
        L[j,0:j] = forwardsubs(L[0:j,0:j],B[j,0:j])
        L[j,j] = np.sqrt(B[j,j] - LA.norm(L[j,0:j])**2)
    return L


def ls_by_chol(A, b):
    """
    Solves the linear least squares problem using Cholesky decomposition.

    Parameters:
    A (numpy.ndarray): The coefficient matrix.
    b (numpy.ndarray): The dependent variable vector.

    Returns:
    numpy.ndarray: The solution vector x that minimizes the squared Euclidean norm ||Ax - b||^2.
    """
    L = cholesky(A.T @ A)
    z = forwardsubs(L, A.T @ b)
    return backsubs(L.T, z)


def gramschmidt(A):
    """
    Performs the Gram-Schmidt process on the given matrix A.

    Parameters:
    A (numpy.ndarray): The input matrix of shape (n, m).

    Returns:
    Q (numpy.ndarray): The orthogonal matrix Q of shape (n, m).
    R (numpy.ndarray): The upper triangular matrix R of shape (m, m).
    """

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
    """
    Performs the Householder transformation on a matrix A and a vector b.

    Parameters:
    A (numpy.ndarray): The input matrix of shape (n, m).
    b (numpy.ndarray): The input vector of shape (n,).

    Returns:
    R (numpy.ndarray): The transformed matrix R of shape (m, m).
    Qtb (numpy.ndarray): The transformed vector Qtb of shape (m,).
    """
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


def ls_by_qr(A, b):
    """
    Solves a linear system of equations using QR decomposition.

    Parameters:
    A (numpy.ndarray): The coefficient matrix of the linear system.
    b (numpy.ndarray): The right-hand side vector of the linear system.

    Returns:
    numpy.ndarray: The solution vector x that satisfies Ax = b.
    """
    Q, R = gramschmidt(A)
    return backsubs(R, Q.T @ b)



# Spectral and SVD methods

def topsing(A, maxiter=10):
    """
    Compute the top singular triplets of a matrix A.

    Parameters:
    A (ndarray): Input matrix.
    maxiter (int): Maximum number of iterations for power iteration method. Default is 10.

    Returns:
    u (ndarray): Left singular vector corresponding to the largest singular value.
    s (float): Largest singular value.
    v (ndarray): Right singular vector corresponding to the largest singular value.
    """
    x = np.random.normal(0, 1, np.shape(A)[1])
    B = A.T @ A
    for _ in range(maxiter):
        x = B @ x
    v = x / LA.norm(x)
    s = LA.norm(A @ v)
    u = A @ v / s
    return u, s, v


def svd(A, l, maxiter=100):
    """
    Perform Singular Value Decomposition (SVD) on a matrix A.

    Parameters:
    A (ndarray): Input matrix of shape (m, n).
    l (int): Number of singular values to compute.
    maxiter (int, optional): Maximum number of iterations for the algorithm. Default is 100.

    Returns:
    U (ndarray): Left singular vectors of shape (m, l).
    S (list): Singular values.
    V (ndarray): Right singular vectors of shape (n, l).
    """
    V = rng.normal(0,1,(np.size(A,1),l))
    for _ in range(maxiter):
        W = A @ V
        Z = A.T @ W
        V, R = gramschmidt(Z)
    W = A @ V
    S = [LA.norm(W[:, i]) for i in range(np.size(W,1))]
    U = np.stack([W[:,i]/S[i] for i in range(np.size(W,1))],axis=-1)
    return U, S, V



# Data simulation

def one_cluster(d, n, w):
    """
    Generate a single cluster of data points.

    Parameters:
    - d (int): The dimensionality of the data points.
    - n (int): The number of data points to generate.
    - w (float): The weight of the first dimension in each data point.

    Returns:
    - X (ndarray): An array of shape (n, d) containing the generated data points.
    """
    X = np.stack(
        [np.concatenate(([w], np.zeros(d-1))) + rng.normal(0,1,d) for _ in range(n)]
    )
    return X


def two_clusters(d, n, w):
    """
    Generate two clusters of data points.

    Parameters:
    - d (int): The dimensionality of the data points.
    - n (int): The number of data points in each cluster.
    - w (float): The distance between the two clusters.

    Returns:
    - X1 (list): The data points in the first cluster.
    - X2 (list): The data points in the second cluster.
    """
    X1 = one_cluster(d, n, -w)
    X2 = one_cluster(d, n, w)
    return X1, X2


# Spectral graph theory algorithms

def cut_ratio(A, order, k):
    """
    Calculates the cut ratio of a graph given its adjacency matrix and a vertex order.

    Parameters:
    A (numpy.ndarray): The adjacency matrix of the graph.
    order (list): The order of vertices in the cut.
    k (int): The index of the last vertex in the cut.

    Returns:
    float: The cut ratio of the graph.
    """
    n = A.shape[0] # number of vertices
    edge_boundary = 0 # initialize size of edge boundary 
    for i in range(k+1): # for all vertices before cut
        for j in range(k+1,n): # for all vertices after cut
            edge_boundary += A[order[i],order[j]] # add one if {i,j} in E
    denominator = np.minimum(k+1, n-k-1)
    return edge_boundary/denominator


def spectral_cut2(A):
    """
    Perform spectral cut on a graph represented by its adjacency matrix.

    Parameters:
    A (numpy.ndarray): The adjacency matrix of the graph.

    Returns:
    tuple: A tuple containing two numpy arrays representing the two partitions of the graph.

    """
    n = A.shape[0] # number of vertices
    
    # laplacian
    degrees = A.sum(axis=1)
    D = np.diag(degrees)
    L = D - A

    # spectral decomposition
    w, v = LA.eigh(L) 
    order = np.argsort(v[:,np.argsort(w)[1]]) # index of entries in increasing order
    
    # cut ratios
    phi = np.zeros(n-1) # initialize cut ratios
    for k in range(n-1):
        phi[k] = cut_ratio(A, order, k)
    imin = np.argmin(phi) # find best cut ratio
    return order[0:imin+1], order[imin+1:n]


def viz_cut(G, s, layout):
    """
    Visualizes a graph with a highlighted cut.

    Parameters:
    - G: NetworkX graph object
        The graph to be visualized.
    - s: int
        The index of the node to be highlighted.
    - layout: function
        A function that computes the layout of the graph.

    Returns:
    None
    """
    n = G.number_of_nodes()
    assign = np.ones(n)
    assign[s] = 2
    nx.draw_networkx(G, node_color=assign, pos=layout(G), with_labels=False)



# Optimization algorithms
    
def desc_update(grad_f, x, alpha):
    """
    Performs a gradient descent update on the input variable x.

    Parameters:
    - grad_f: The gradient of the function f at x.
    - x: The current value of the variable.
    - alpha: The learning rate or step size for the update.

    Returns:
    - The updated value of x after performing the gradient descent update.
    """
    return x - alpha*grad_f(x)


def gd(f, grad_f, x0, alpha=1e-3, niters=int(1e6)):
    """
    Performs gradient descent optimization to minimize a given function.

    Parameters:
    f (function): The objective function to be minimized.
    grad_f (function): The gradient function of the objective function.
    x0 (float or array-like): The initial point for optimization.
    alpha (float, optional): The learning rate or step size. Defaults to 1e-3.
    niters (int, optional): The maximum number of iterations. Defaults to 1e6.

    Returns:
    tuple: A tuple containing the optimized point and the value of the objective function at that point.
    """

    xk = x0
    for _ in range(niters):
        xk = desc_update(grad_f, xk, alpha)

    return xk, f(xk)




