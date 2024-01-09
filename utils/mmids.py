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




# Markov chains algorithms

import numpy as np

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


def transition_from_digraph(G):
    """
    Compute the transition matrix from a directed graph.

    Parameters:
    - G: NetworkX DiGraph object representing the directed graph.

    Returns:
    - numpy.ndarray: The transition matrix of the directed graph.
    """
    n = G.number_of_nodes()
    invD = np.zeros((n,n))
    for i in range(n):
        invD[i,i] = 1 / G.out_degree(i)
    A = nx.adjacency_matrix(G).toarray()
    return invD @ A


import numpy as np
import networkx as nx

def transition_from_graph(G):
    """
    Compute the transition matrix from a graph.

    Parameters:
    - G: NetworkX graph object

    Returns:
    - numpy.ndarray: The transition matrix of the graph
    """
    n = G.number_of_nodes()
    invD = np.zeros((n,n))
    for i in range(n):
        invD[i,i] = 1 / G.degree(i)
    A = nx.adjacency_matrix(G).toarray()
    return invD @ A



def pagerank(M, num_iterations: int = 100, d: float = 0.85):
    """
    Parameters
    ----------
    M : numpy array
        adjacency matrix transposed where M_i,j represents 
        the link from 'j' to 'i', such that for all 'j' sum(i, M_i,j) = 1
    num_iterations : int, optional
        number of iterations, by default 100
    d : float, optional
        damping factor, by default 0.85
    
    Returns
    -------
    numpy array
        a vector of ranks such that v_i is the i-th rank from [0, 1],
        v sums to 1
    """
    n = M.shape[1]
    v = np.ones(n)
    v = v / n
    for _ in range(num_iterations):
        v = d * M @ v + (1-d) * np.ones(n)/n
    return v



def pagerank_from_adjacency(A, num_iter: int = 100, d: float = 0.85):
    """
    Parameters
    ----------
    A : numpy array
        adjacency matrix where M_i,j represents 
        the link from 'i' to 'j'
    num_iterations : int, optional
        number of iterations, by default 100
    d : float, optional
        damping factor, by default 0.85
    
    Returns
    -------
    numpy array
        a vector of ranks such that v_i is the i-th rank from [0, 1],
        v sums to 1
    """
    n = A.shape[0]
    v = np.ones(n)
    out_deg = A @ v
    v = v / n
    for _ in range(num_iter):
        v = d * A.T @ np.divide(v, out_deg, out=np.zeros_like(v), where=out_deg!=0) + (1-d) * np.ones(n)/n
    return v






# Linear Gaussian models


def lgSamplePath(ss, os, F, H, Q, R, x_0, T):
    """
    Generate a sample path from a linear Gaussian state space model.

    Parameters:
    ss (int): The dimension of the state vector.
    os (int): The dimension of the observation vector.
    F (ndarray): The state transition matrix of shape (ss, ss).
    H (ndarray): The observation matrix of shape (os, ss).
    Q (ndarray): The state noise covariance matrix of shape (ss, ss).
    R (ndarray): The observation noise covariance matrix of shape (os, os).
    x_0 (ndarray): The initial state vector of shape (ss,).
    T (int): The number of time steps.

    Returns:
    x (ndarray): The generated state path of shape (ss, T).
    y (ndarray): The generated observation path of shape (os, T).
    """
    x = np.zeros((ss,T)) 
    y = np.zeros((os,T))
    x[:,0] = x_0
    ey = np.zeros(os)
    ey = rng.multivariate_normal(np.zeros(os),R) 
    y[:,0] = H @ x[:,0] + ey
    
    for t in range(1,T):
        ex = np.zeros(ss)
        ex = rng.multivariate_normal(np.zeros(ss),Q) # noise on x_t
        x[:,t] = F @ x[:,t-1] + ex
        ey = np.zeros(os)
        ey = rng.multivariate_normal(np.zeros(os),R) # noise on y_t
        y[:,t] = H @ x[:,t] + ey
    
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
        e_t = y_t - C @ mu_pred  # error at time t
        S = C @ Sig_pred @ C.T + R
        Sinv = LA.inv(S)
        K = Sig_pred @ C.T @ Sinv  # Kalman gain matrix
        mu_new = mu_pred + K @ e_t
        Sig_new = (np.diag(np.ones(ss)) - K @ C) @ Sig_pred
        return mu_new, Sig_new
    


import numpy as np

def kalmanFilter(ss, os, y, A, C, Q, R, init_mu, init_Sig, T):
    """
    Apply the Kalman filter algorithm to estimate the hidden states of a linear dynamical system.

    Parameters:
    - ss (int): Number of hidden states.
    - os (int): Number of observed states.
    - y (ndarray): Observations of shape (os, T), where T is the number of time steps.
    - A (ndarray): State transition matrix of shape (ss, ss).
    - C (ndarray): Observation matrix of shape (os, ss).
    - Q (ndarray): Process noise covariance matrix of shape (ss, ss).
    - R (ndarray): Observation noise covariance matrix of shape (os, os).
    - init_mu (ndarray): Initial state mean of shape (ss,).
    - init_Sig (ndarray): Initial state covariance matrix of shape (ss, ss).
    - T (int): Number of time steps.

    Returns:
    - mu (ndarray): Estimated state means of shape (ss, T).
    - Sig (ndarray): Estimated state covariance matrices of shape (ss, ss, T).
    """
    mu = np.zeros((ss, T))
    Sig = np.zeros((ss, ss, T))
    mu[:,0] = init_mu
    Sig[:,:,0] = init_Sig

    for t in range(1,T):
        mu[:,t], Sig[:,:,t] = kalmanUpdate(ss, A, C, Q, R, y[:,t], mu[:,t-1], Sig[:,:,t-1])

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
        
    K, M = N_km.shape
    
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



