from scipy.special import psi, gamma
from scipy.spatial import KDTree, cKDTree

from numpy import floor, sqrt, concatenate, ones, sort, mean, log, absolute,\
                  exp, pi, sum, max
from numpy.random import permutation, choice
from numpy import pi, cumsum, hstack, zeros, sum, ix_, mean, newaxis, \
                  sqrt, dot, median, exp, min, floor, log, eye, absolute, \
                  array, max, any, place, inf, isinf, where, diag
import numpy as np
from numpy.random import rand
from numpy import ones
from numpy import zeros, mod, array


NOISE = 1e-10

class ExceptionCompSubspaceDims(Exception):

    def __str__(self):
        return 'The subspace dimensions are not compatible with y!'


class VerCompSubspaceDims(object):
    def verification_compatible_subspace_dimensions(self, y, ds):

        if y.shape[1] != sum(ds):
            raise ExceptionCompSubspaceDims()

def knn_distances(y, q, y_equals_to_q, knn_method='cKDTree', knn_k=3,
                  knn_eps=0, knn_p=2):
    
    if knn_method == 'cKDTree':
        tree = cKDTree(y)    
    elif knn_method == 'KDTree':
        tree = KDTree(y)

    if y_equals_to_q:
        if knn_k+1 > y.shape[0]:
            raise Exception("'knn_k' + 1 <= 'num_of_samples in y' " + 
                            "is not satisfied!")
                            
        # distances, indices: |q| x (knn_k+1):                                
        distances, indices = tree.query(q, k=knn_k+1, eps=knn_eps, p=knn_p)
        
        # exclude the points themselves => distances, indices: |q| x knn_k:
        distances, indices = distances[:, 1:], indices[:, 1:]
    else: 
        if knn_k > y.shape[0]:
            raise Exception("'knn_k' <= 'num_of_samples in y' " + 
                            "is not satisfied!")
                            
        # distances, indices: |q| x knn_k:                            
        distances, indices = tree.query(q, k=knn_k, eps=knn_eps, p=knn_p) 
        
    return distances, indices


def volume_of_the_unit_ball(d):
    vol = pi**(d/2) / gamma(d/2+1)  # = 2 * pi^(d/2) / ( d*gamma(d/2) )
    
    return vol


class InitX(object):
    def __init__(self, mult=True):
        self.mult = mult

    def __str__(self):
        return ''.join((self.__class__.__name__, ' -> ',
                        str(self.__dict__)))

class InitKnnK(InitX):

    def __init__(self, mult=True, knn_method='cKDTree', k=3, eps=0):
        super().__init__(mult=mult)

        # other attributes:
        self.knn_method, self.k, self.eps = knn_method, k, eps

class BHShannon_KnnK(InitKnnK):
    def estimation_no_chunk(self, y):
        num_of_samples, dim = y.shape
        distances_yy = knn_distances(y, y, True, self.knn_method, self.k,
                                     self.eps, 2)[0]
        v = volume_of_the_unit_ball(dim)
        h = log(num_of_samples - 1) - psi(self.k) + log(v) + \
            dim * sum(np.log(np.maximum(distances_yy[:, self.k-1], NOISE))) / num_of_samples

        return h

    def estimation_chunk(self, y, M):
        n_chunks = int(y.shape[0] / M)
        chunks = np.array_split(np.random.permutation(y), n_chunks)
        ent = 0
        for chunk in chunks:
            ent += self.estimation_no_chunk(chunk)

        ent /= n_chunks
        return ent

    def estimation(self, y, chunk = None):
        if chunk is None:
            return self.estimation_no_chunk(y)
        else:
            return self.estimation_chunk(y, M = chunk)


class MIShannon_HS(InitX, VerCompSubspaceDims):
    def __init__(self, mult=True, k = 1):
        super().__init__(mult=mult)
        
        self.shannon_co = BHShannon_KnnK(k = k)
        
    def estimation(self, y, ds, chunk = None):
        self.verification_compatible_subspace_dimensions(y, ds)

        # I = - H([y^1,...,y^M]):
        i = -self.shannon_co.estimation(y, chunk = chunk)

        # I = I + \sum_{m=1}^M H(y^m):
        idx_start = 0 
        for k in range(len(ds)):
            dim_k = ds[k]
            idx_stop = idx_start + dim_k
            i += self.shannon_co.estimation(y[:, idx_start:idx_stop], chunk = chunk)
            idx_start = idx_stop    

        return i