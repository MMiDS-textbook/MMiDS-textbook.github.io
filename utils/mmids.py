# Description: Utility functions for MMIDS


# Libraries

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import networkx as nx
seed = 535
rng = np.random.default_rng(seed)


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


def pca(X, l, maxiter=100):
    """
    Perform Principal Component Analysis (PCA) on the input data.

    Parameters:
    X (numpy.ndarray): Input data matrix of shape (n_samples, n_features).
    l (int): Number of principal components to keep.
    maxiter (int, optional): Maximum number of iterations for the SVD algorithm. Default is 100.

    Returns:
    numpy.ndarray: Transformed data matrix of shape (n_samples, l), where l is the number of principal components.

    """
    mean = np.mean(X, axis=0)  # Compute mean of each column
    Y = X - mean # Mean center each column
    U, S, V = svd(Y, l, maxiter=maxiter)
    return U[:,:l] @ np.diag(S[:l])





# Data simulation

def one_cluster(d, n, w):
    """
    DEPRECATED: See spherical_gaussian
    
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
    DEPRECATED: See two_separated_clusters
    
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





def spherical_gaussian(d, n, mu, sig):
    """
    Generate samples from a spherical Gaussian distribution.

    Parameters:
    - d (int): The dimensionality of the samples.
    - n (int): The number of samples to generate.
    - mu (float): The mean of the distribution.
    - sig (float): The standard deviation of the distribution.

    Returns:
    - X (ndarray): An array of shape (n, d) containing the generated samples.
    """

    X = mu + sig * rng.normal(0,1,(n,d))

    return X




def gmm2spherical(d, n, phi0, phi1, mu0, sig0, mu1, sig1):
    """
    Generate samples from a Gaussian Mixture Model (GMM) with spherical Gaussian components.
    
    Parameters:
    - d (int): The dimensionality of the samples.
    - n (int): The number of samples to generate.
    - phi0 (float): The weight of the first component.
    - phi1 (float): The weight of the second component.
    - mu0 (ndarray): The mean vector of the first component.
    - sig0 (float): The standard deviation of the first component.
    - mu1 (ndarray): The mean vector of the second component.
    - sig1 (float): The standard deviation of the second component.
    
    Returns:
    - X (ndarray): The generated samples, with shape (n, d).
    """
    
    # merge components into matrices
    phi = np.stack((phi0, phi1))
    mu = np.stack((mu0, mu1))
    sig = np.stack((sig0,sig1))
    
    # initialization
    X = np.zeros((n,d))
    
    # choose components of each data point, then generate samples
    component = rng.choice(2, size=n, p=phi)
    for i in range(n):
        X[i,:] = spherical_gaussian(
            d, 1, mu[component[i],:], sig[component[i]])
    
    return X





def gmm2(d, n, phi0, phi1, mu0, sigma0, mu1, sigma1):
    """
    Generate samples from a Gaussian Mixture Model (GMM) with 2 components.
    
    Parameters:
    - d (int): The dimensionality of the samples.
    - n (int): The number of samples to generate.
    - phi0 (float): The mixing coefficient for component 0.
    - phi1 (float): The mixing coefficient for component 1.
    - mu0 (ndarray): The mean vector for component 0.
    - sigma0 (ndarray): The covariance matrix for component 0.
    - mu1 (ndarray): The mean vector for component 1.
    - sigma1 (ndarray): The covariance matrix for component 1.
    
    Returns:
    - X (ndarray): The generated samples, with shape (n, d).
    """
    
    # merge components into tensors
    phi = np.stack((phi0, phi1))
    mu = np.stack((mu0, mu1))
    sigma = np.stack((sigma0,sigma1))
    
    # initialization
    X = np.zeros((n,d))
    
    # choose components of each data point, then generate samples
    component = rng.choice(2, size=n, p=phi)
    for i in range(n):
        X[i,:] = rng.multivariate_normal(
            mu[component[i],:],
            sigma[component[i],:,:])
    
    return X



def two_mixed_clusters(d, n, w):
    """
    Generate a dataset with two mixed clusters.

    Parameters:
    - d (int): The dimensionality of the dataset.
    - n (int): The number of data points to generate.
    - w (float): The separation between the two clusters.

    Returns:
    - ndarray: The generated dataset with shape (n, d).
    """
    
    # set parameters
    phi0 = 0.5
    phi1 = 0.5
    mu0 = np.concatenate(([w], np.zeros(d-1)))
    mu1 = np.concatenate(([-w], np.zeros(d-1)))
    sig0 = 1
    sig1 = 1
    
    return gmm2spherical(
        d, n, phi0, phi1, mu0, sig0, mu1, sig1)




def two_separated_clusters(d, n, w):
    """
    Generate two separated clusters of samples in d-dimensional space.
    
    Parameters:
        d (int): The dimensionality of the samples.
        n (int): The number of samples to generate for each cluster.
        w (float): The separation between the two clusters.
        
    Returns:
        tuple: A tuple containing two arrays, X0 and X1, representing the samples
               from the first and second clusters respectively.
    """
    
    # set parameters
    mu0 = np.concatenate(([w], np.zeros(d-1)))
    mu1 = np.concatenate(([-w], np.zeros(d-1)))
    sig0 = 1
    sig1 = 1
    
    # generate samples
    X0 = spherical_gaussian(d, n, mu0, sig0)
    X1 = spherical_gaussian(d, n, mu1, sig1)
    
    return X0, X1





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




# Markov chains algorithms

def SamplePath(mu, P, T):
    """
    Generate a sample path from a Markov chain.

    Parameters:
    mu (numpy.ndarray): The initial distribution of the Markov chain.
    P (numpy.ndarray): The transition matrix of the Markov chain.
    T (int): The length of the sample path.

    Returns:
    numpy.ndarray: The generated sample path.

    """
    n = mu.shape[0] # size of state space
    X = np.zeros(T+1) # initialization of sample path
    for i in range(T+1):
        if i == 0: # initial distribution
            X[i] = np.random.choice(a=np.arange(start=1,stop=n+1),p=mu)
        else: # next state is chosen from current state row
            X[i] = np.random.choice(a=np.arange(start=1,stop=n+1),p=P[int(X[i-1]-1),:])
    return X


def transition_from_adjacency(A):
    """
    Compute the transition matrix from an adjacency matrix.

    Parameters:
    A (numpy.ndarray): The adjacency matrix.

    Returns:
    numpy.ndarray: The transition matrix.

    """
    n = A.shape[0]
    sinks = (A @ np.ones(n)) == 0.
    P = A.copy()
    np.fill_diagonal(P, sinks)
    out_deg = P @ np.ones(n)
    P = P / out_deg[:, np.newaxis]
    return P


def add_damping(P, alpha, mu):
    """
    Adds damping to a matrix P using the given damping factor alpha and damping matrix mu.

    Parameters:
    P (numpy.ndarray): The matrix to which damping is applied.
    alpha (float): The damping factor, ranging from 0 to 1.
    mu (numpy.ndarray): The damping matrix.

    Returns:
    numpy.ndarray: The damped matrix Q, calculated as alpha * P + (1-alpha) * mu.
    """
    Q = alpha * P + (1-alpha) * mu
    return Q


def pagerank(A, alpha=0.85, max_iter=100):
    """
    Calculate the PageRank scores for a given adjacency matrix.

    Parameters:
    - A: numpy.ndarray
        The adjacency matrix representing the graph.
    - alpha: float, optional
        The damping factor, which determines the probability of following a link.
        Default is 0.85.
    - max_iter: int, optional
        The maximum number of iterations for the PageRank algorithm.
        Default is 100.

    Returns:
    - v: numpy.ndarray
        The PageRank scores for each node in the graph.
    """
    n = A.shape[0]
    mu = np.ones(n)/n
    P = transition_from_adjacency(A)
    Q = add_damping(P, alpha, mu)
    v = mu
    for _ in range(max_iter):
        v = Q.T @ v
    return v



def ppr(A, mu, alpha=0.85, max_iter=100):
    """
    Calculates the Personalized PageRank (PPR) vector for a given adjacency matrix.

    Parameters:
    A (numpy.ndarray): The adjacency matrix representing the graph.
    mu (float): The teleportation probability.
    alpha (float, optional): The damping factor. Default is 0.85.
    max_iter (int, optional): The maximum number of iterations. Default is 100.

    Returns:
    numpy.ndarray: The PPR vector.

    """
    n = A.shape[0]
    P = transition_from_adjacency(A)
    Q = add_damping(P, alpha, mu)
    v = mu
    for _ in range(max_iter):
        v = Q.T @ v
    return v


    








# Linear Gaussian models


def lgSamplePath(ss, os, F, H, Q, R, init_mu, init_Sig, T):
    """
    Generate a sample path from a linear Gaussian state-space model.

    Parameters:
    ss (int): The number of state variables.
    os (int): The number of observation variables.
    F (ndarray): The state transition matrix of shape (ss, ss).
    H (ndarray): The observation matrix of shape (os, ss).
    Q (ndarray): The state noise covariance matrix of shape (ss, ss).
    R (ndarray): The observation noise covariance matrix of shape (os, os).
    init_mu (ndarray): The initial state mean vector of shape (ss,).
    init_Sig (ndarray): The initial state covariance matrix of shape (ss, ss).
    T (int): The number of time steps.

    Returns:
    x (ndarray): The generated state path of shape (ss, T).
    y (ndarray): The generated observation path of shape (os, T).
    """
    x = np.zeros((ss, T)) 
    y = np.zeros((os, T))

    x[:, 0] = np.random.multivariate_normal(init_mu, init_Sig)
    for t in range(1, T):
        x[:, t] = np.random.multivariate_normal(F @ x[:, t-1], Q)  # noise on x_t
        y[:, t] = np.random.multivariate_normal(H @ x[:, t], R)  # noise on y_t
    
    return x, y



def kalmanUpdate(ss, A, C, Q, R, y_t, mu_prev, Sig_prev):
    """
    Performs the Kalman update step.

    Args:
        ss (int): State size.
        A (ndarray): State transition matrix.
        C (ndarray): Observation matrix.
        Q (ndarray): Process noise covariance matrix.
        R (ndarray): Measurement noise covariance matrix.
        y_t (ndarray): Measurement vector at time t.
        mu_prev (ndarray): Previous state estimate.
        Sig_prev (ndarray): Previous state covariance matrix.

    Returns:
        tuple: Updated state estimate (mu_new) and state covariance matrix (Sig_new).
    """
    mu_pred = A @ mu_prev
    Sig_pred = A @ Sig_prev @ A.T + Q
    if np.isnan(y_t[0]) or np.isnan(y_t[1]):
        return mu_pred, Sig_pred
    else:
        e_t = y_t - C @ mu_pred # error at time t
        S = C @ Sig_pred @ C.T + R
        Sinv = LA.inv(S)
        K = Sig_pred @ C.T @ Sinv # Kalman gain matrix
        mu_new = mu_pred + K @ e_t
        Sig_new = (np.diag(np.ones(ss)) - K @ C) @ Sig_pred
        return mu_new, Sig_new


    


def kalmanFilter(ss, os, y, A, C, Q, R, init_mu, init_Sig, T):
    """
    Applies the Kalman filter algorithm to estimate the hidden states of a linear dynamical system.

    Parameters:
    ss (int): The number of hidden states.
    os (int): The number of observed states.
    y (ndarray): The observed states at each time step, shape (os, T).
    A (ndarray): The state transition matrix, shape (ss, ss).
    C (ndarray): The observation matrix, shape (os, ss).
    Q (ndarray): The process noise covariance matrix, shape (ss, ss).
    R (ndarray): The observation noise covariance matrix, shape (os, os).
    init_mu (ndarray): The initial mean of the hidden states, shape (ss,).
    init_Sig (ndarray): The initial covariance matrix of the hidden states, shape (ss, ss).
    T (int): The number of time steps.

    Returns:
    mu (ndarray): The estimated means of the hidden states at each time step, shape (ss, T).
    Sig (ndarray): The estimated covariance matrices of the hidden states at each time step, shape (ss, ss, T).
    """
    mu = np.zeros((ss, T))
    Sig = np.zeros((ss, ss, T))
    mu[:,0] = init_mu
    Sig[:,:,0] = init_Sig

    for t in range(1,T):
        mu[:,t], Sig[:,:,t] = kalmanUpdate(ss, A, C, Q, R, 
                                           y[:,t], mu[:,t-1], 
                                           Sig[:,:,t-1])

    return mu, Sig





# Probabilistic models

def nb_fit_table(N_km, alpha=1., beta=1.):
    """
    Fits a Naive Bayes model to a contingency table.

    Parameters:
    - N_km (ndarray): Contingency table of shape (K, M) where K is the number of classes and M is the number of features.
    - alpha (float): Smoothing parameter for the class probabilities. Default is 1.
    - beta (float): Smoothing parameter for the feature probabilities. Default is 1.

    Returns:
    - pi_k (ndarray): Maximum likelihood estimates for the class probabilities of shape (K,).
    - p_km (ndarray): Maximum likelihood estimates for the feature probabilities of shape (K, M).
    """
    
    K, M = N_km.shape
    N_k = np.sum(N_km,axis=-1)
    N = np.sum(N_k)
    
    # MLE for pi_k's
    pi_k = (N_k+alpha) / (N+K*alpha)
    
    # MLE for p_km's
    p_km = (N_km+beta) / (N_k[:,None]+2*beta)

    return pi_k, p_km


def nb_predict(pi_k, p_km, x, label_set):
    """
    Predicts the label for a given input using the Naive Bayes classifier.

    Parameters:
    - pi_k (list): The prior probabilities for each class.
    - p_km (ndarray): The conditional probabilities for each feature given each class.
    - x (ndarray): The input features.
    - label_set (list): The set of possible labels.

    Returns:
    - predicted_label: The predicted label for the input.
    """
   
    K = len(pi_k)
    
    # Computing the score for each k
    score_k = np.zeros(K)
    for k in range(K):
       
        score_k[k] += - np.log(pi_k[k])
        score_k[k] += - np.sum(x * np.log(p_km[k,:]) + (1 - x)*np.log(1 - p_km[k,:]))
    
    # Computing the minimum
    argmin = np.argmin(score_k, axis=0)
    minscr = np.max(score_k, axis=0)

    return label_set[argmin]



def responsibility(pi_k, p_km, x):
    """
    Compute the responsibilities for each component in a mixture model.

    Parameters:
    - pi_k (array-like): The mixing coefficients for each component.
    - p_km (array-like): The conditional probabilities for each component.
    - x (array-like): The observed data.

    Returns:
    - r_k (array-like): The responsibilities for each component.

    """
    K = len(pi_k)
        
    # Computing the score for each k
    score_k = np.zeros(K)
    for k in range(K):
        score_k[k] += - np.log(pi_k[k])
        score_k[k] += - np.sum(x*np.log(p_km[k,:]) + (1 - x)*np.log(1 - p_km[k,:]))
    
    # Computing responsibilities for each k
    r_k = np.exp(-score_k)/(np.sum(np.exp(-score_k)))
        
    return r_k


def update_parameters(eta_km, eta_k, eta, alpha, beta):
    """
    Update the parameters for the MMiDS model.

    Parameters:
    - eta_km: numpy.ndarray, shape (K, M)
        The count matrix of the number of times each keyword m is assigned to topic k.
    - eta_k: numpy.ndarray, shape (K,)
        The count vector of the number of times each topic k is assigned to any document.
    - eta: float
        The total count of topic assignments to any document.
    - alpha: float
        The hyperparameter for the Dirichlet prior on the topic distribution.
    - beta: float
        The hyperparameter for the Dirichlet prior on the keyword distribution.

    Returns:
    - pi_k: numpy.ndarray, shape (K,)
        The maximum likelihood estimate of the topic distribution.
    - p_km: numpy.ndarray, shape (K, M)
        The maximum likelihood estimate of the keyword distribution.
    """
        
    K = len(eta_k)
    
    # MLE for pi_k's
    pi_k = (eta_k+alpha) / (eta+K*alpha)
    
    # MLE for p_km's
    p_km = (eta_km+beta) / (eta_k[:,None]+2*beta)

    return pi_k, p_km



def em_bern(X, K, pi_0, p_0, maxiters=10, alpha=0., beta=0.):
    """
    Expectation-Maximization algorithm for estimating parameters of a Bernoulli Mixture Model.

    Parameters:
    - X: numpy.ndarray
        Input data matrix of shape (n, M), where n is the number of samples and M is the number of features.
    - K: int
        Number of mixture components.
    - pi_0: numpy.ndarray
        Initial guess for the mixing coefficients of shape (K,).
    - p_0: numpy.ndarray
        Initial guess for the Bernoulli parameters of shape (K, M).
    - maxiters: int, optional
        Maximum number of iterations for the EM algorithm. Default is 10.
    - alpha: float, optional
        Smoothing parameter for the mixing coefficients. Default is 0.
    - beta: float, optional
        Smoothing parameter for the Bernoulli parameters. Default is 0.

    Returns:
    - pi_k: numpy.ndarray
        Estimated mixing coefficients of shape (K,).
    - p_km: numpy.ndarray
        Estimated Bernoulli parameters of shape (K, M).
    """
    
    n, M = X.shape
    pi_k = pi_0
    p_km = p_0
    
    for _ in range(maxiters):
    
        # E Step
        r_ki = np.zeros((K, n))
        for i in range(n):
            r_ki[:, i] = responsibility(pi_k, p_km, X[i, :])
        
        # M Step     
        eta_km = np.zeros((K, M))
        eta_k = np.sum(r_ki, axis=-1)
        eta = np.sum(eta_k)
        for k in range(K):
            for m in range(M):
                eta_km[k, m] = np.sum(X[:, m] * r_ki[k, :]) 
        pi_k, p_km = update_parameters(eta_km, eta_k, eta, alpha, beta)
        
    return pi_k, p_km





def hard_responsibility(pi_k, p_km, x):
    """
    Compute the hard responsibilities for each cluster based on the given parameters.

    Parameters:
    - pi_k (numpy.ndarray): The probabilities of each cluster.
    - p_km (numpy.ndarray): The probabilities of each feature given each cluster.
    - x (numpy.ndarray): The observed data.

    Returns:
    - r_k (numpy.ndarray): The hard responsibilities for each cluster.
    """

    K = len(pi_k)
        
    # Computing the score for each k
    score_k = np.zeros(K)
    for k in range(K):
        score_k[k] += - np.log(pi_k[k])
        score_k[k] += - np.sum(x*np.log(p_km[k,:]) + (1 - x)*np.log(1 - p_km[k,:]))
    
    # Computing responsibilities for each k
    argmin = np.argmin(score_k, axis=0)
    r_k = np.zeros(K)
    r_k[argmin] = 1

    return r_k


def hard_em_bern(X, K, pi_0, p_0, maxiters=10, alpha=0., beta=0.):
    """
    Perform hard expectation-maximization (EM) algorithm for Bernoulli mixture model.

    Parameters:
    - X: numpy.ndarray
        Input data matrix of shape (n, M), where n is the number of samples and M is the number of features.
    - K: int
        Number of mixture components.
    - pi_0: numpy.ndarray
        Initial mixing coefficients of shape (K,).
    - p_0: numpy.ndarray
        Initial Bernoulli parameters of shape (K, M).
    - maxiters: int, optional
        Maximum number of iterations for the EM algorithm. Default is 10.
    - alpha: float, optional
        Hyperparameter for the Dirichlet prior on mixing coefficients. Default is 0.
    - beta: float, optional
        Hyperparameter for the Beta prior on Bernoulli parameters. Default is 0.

    Returns:
    - pi_k: numpy.ndarray
        Estimated mixing coefficients after the EM algorithm, of shape (K,).
    - p_km: numpy.ndarray
        Estimated Bernoulli parameters after the EM algorithm, of shape (K, M).
    """
    
    n, M = X.shape
    pi_k = pi_0
    p_km = p_0
    
    for _ in range(maxiters):
    
        # E Step
        r_ki = np.zeros((K, n))
        for i in range(n):
            r_ki[:, i] = hard_responsibility(pi_k, p_km, X[i, :])
        
        # M Step     
        eta_km = np.zeros((K, M))
        eta_k = np.sum(r_ki, axis=-1)
        eta = np.sum(eta_k)
        for k in range(K):
            for m in range(M):
                eta_km[k, m] = np.sum(X[:, m] * r_ki[k, :]) 
        pi_k, p_km = update_parameters(eta_km, eta_k, eta, alpha, beta)
        
    return pi_k, p_km





# Gibbs sampling for RBMs


def sigmoid(z):
    """
    Compute the sigmoid function.

    Parameters:
    z (float or array-like): The input value(s) to the sigmoid function.

    Returns:
    float or array-like: The output value(s) of the sigmoid function.

    """
    return 1 / (1 + np.exp(-z))



def rbm_mean_hidden(v, W, c):
    """
    Computes the mean activation of hidden units in a Restricted Boltzmann Machine (RBM).

    Parameters:
    v (numpy.ndarray): Input vector of visible units.
    W (numpy.ndarray): Weight matrix connecting visible and hidden units.
    c (numpy.ndarray): Bias vector for hidden units.

    Returns:
    numpy.ndarray: Mean activation of hidden units.

    """
    return sigmoid(W @ v + c.reshape(len(c),1))


def rbm_mean_visible(h, W, b):
    """
    Computes the mean of the visible units in a Restricted Boltzmann Machine (RBM).

    Parameters:
    h (numpy.ndarray): Hidden units values.
    W (numpy.ndarray): Weight matrix connecting hidden and visible units.
    b (numpy.ndarray): Bias vector for the visible units.

    Returns:
    numpy.ndarray: Mean of the visible units.

    """
    return sigmoid(W.T @ h + b.reshape(len(b),1))



def rbm_gibbs_update(v, W, b, c):
    """
    Performs one Gibbs sampling update step for a Restricted Boltzmann Machine (RBM).

    Args:
        v (ndarray): Visible units of the RBM.
        W (ndarray): Weight matrix connecting visible and hidden units.
        b (ndarray): Bias vector for the visible units.
        c (ndarray): Bias vector for the hidden units.

    Returns:
        ndarray: Updated visible units after one Gibbs sampling step.
    """
    p_hidden = rbm_mean_hidden(v, W, c)
    h = rng.binomial(1, p_hidden, p_hidden.shape)
    p_visible = rbm_mean_visible(h, W, b)
    v = rng.binomial(1, p_visible, p_visible.shape)
    return v


def rbm_gibbs_sampling(k, v_0, W, b, c):
    """
    Perform k steps of Gibbs sampling in a Restricted Boltzmann Machine (RBM).

    Parameters:
    k (int): The number of Gibbs sampling steps to perform.
    v_0 (array-like): The initial visible layer state.
    W (array-like): The weight matrix of the RBM.
    b (array-like): The bias vector of the hidden layer.
    c (array-like): The bias vector of the visible layer.

    Returns:
    array-like: The final visible layer state after k steps of Gibbs sampling.
    """
    counter = 0
    v = v_0
    while counter < k:
        v = rbm_gibbs_update(v, W, b, c)
        counter += 1
    return v


def plot_imgs(z, n_imgs, nx_pixels, ny_pixels):
    """
    Plot a grid of images.

    Parameters:
    - z: numpy array of shape (n_imgs, nx_pixels * ny_pixels)
        The array of images to be plotted.
    - n_imgs: int
        The number of images to be plotted.
    - nx_pixels: int
        The number of pixels in the x-axis of each image.
    - ny_pixels: int
        The number of pixels in the y-axis of each image.
    """
    nx_imgs = np.floor(np.sqrt(n_imgs))
    ny_imgs = np.ceil(np.sqrt(n_imgs))
    plt.figure(figsize=(8, 8))
    for i, comp in enumerate(z):
        plt.subplot(int(nx_imgs), int(ny_imgs), i + 1)
        plt.imshow(comp.reshape((nx_pixels, ny_pixels)), cmap=plt.cm.gray_r)
        plt.xticks([])
        plt.yticks([])
    plt.show()


