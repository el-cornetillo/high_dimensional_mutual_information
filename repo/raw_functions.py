import numpy as np
from scipy.special import psi, gamma
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings('ignore')

NOISE = 1e-10  # Noise added to data to avoid degenerancies
EULER_CONSTANT = -psi(1)  # Euler constant, ~0.577


def nearest_distances(X, k=1):
    ''' Return the nearest distance from each point to his k-th nearest neighboors '''
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X)
    d, _ = knn.kneighbors(X)  # the first nearest neighbor is itself
    return d[:, -1]  # returns the distance to the kth nearest neighbor


def entropy(samples, eps=NOISE):
    ''' Estimate entropy with the K-L formula '''
    N, d = samples.shape
    dvec = nearest_distances(samples)

    return d * np.mean(np.log(np.maximum(dvec, eps))) \
           + np.log(((N - 1) * pow(np.pi, d / 2)) / gamma(1 + d / 2)) \
           + EULER_CONSTANT


def batch_entropy(samples, M=100, eps=NOISE):
    N = samples.shape[0]
    h = 0
    cnt = 0
    samples = np.random.permutation(samples)
    for i in np.arange(0, N, M):
        j = i + M
        if j > N:
            continue
        h += entropy(samples[i:j], eps=eps)
        cnt += 1

    return h / cnt