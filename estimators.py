import numpy as np

from scipy.special import psi, gamma
import scipy.spatial as ss
from scipy.stats import itemfreq

from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings('ignore')

from ite_implementation import *
from utils import *

NOISE = 1e-10 ## Noise added to data to avoid degenerancies
EPS_BALL = 1e-15 ## radius tolerance for the nearest neighboors in the Krashov Method
EULER_CONSTANT = -psi(1) ## Euler constant, ~0.577


class SSD:
    ''' Class to calculate the (noramlized) negative sum of squares '''
    def __init__(self):
        pass

    def _sample(self, img):
        ''' Returns the RGB vectors of each pixel, size N_pixel * d (=3) '''
        return np.array([img[i][j] for i in range(img.shape[0]) for j in range(img.shape[1])]).astype('float')
    
    def get_criterion(self, img1, img2, chunk =None, canal = None):
        # params chunk/canal : for consistency, ignored in this function
        
        x = self._sample(img1)
        y = self._sample(img2)

        return -1 * np.sum((x - y)**2)

    def get_name(self):
        return "SSD"
    
class MI:
    ''' Class to estimate the MI for 2D-images based on the histogram
        method '''
    def __init__(self):
        pass
    
    def _hist(self, img, img2 = None):
        ''' Computes the histogram (or joint histogram) of the data '''
        if img2 is None:
            return np.histogram(img.flatten(), bins = 256)[0] / img.size

        return np.histogram2d(img.flatten(), img2.flatten(), bins = 256)[0].flatten() / img.size

    def _log(self, x):
        if x <=0.:
            return 0
        return x * np.log(x)

    def _entropy(self, img, img2 = None, base = 2):
        _probs = self._hist(img, img2)
        return (-1 * np.sum([self._log(p) for p in _probs])) / np.log(base)
    
    def get_criterion(self, img1, img2, canal = "grayscale", chunk = None):
        # param canal : which strategy to use to transform the image in 2D-space
        # param chunk : for consistency, ignored in this function
        return self._entropy(uint_scale(select_canal(img1, canal))) + self._entropy(uint_scale(select_canal(img2, canal))) \
                    - self._entropy(uint_scale(select_canal(img1, canal)), uint_scale(select_canal(img2, canal)))

    def get_name(self):
        return "MI"
        
        
class BaseKnnMIEstimator:
    ''' Base class for the multi-dimensional MI estimators '''
    def __init__(self, mode):
        self.mode = mode
        pass

    def _joint(self, *kwargs):
        return np.hstack(*kwargs)
    
    def _nearest_distances(self, X, k=1):
        ''' Return the nearest distance from each point to his k-th nearest neighboors '''
        knn = NearestNeighbors(n_neighbors=k+1)
        knn.fit(X)
        d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
        return d[:, -1] # returns the distance to the kth nearest neighbor

    def _psi_avg(self, points, dvec, eps = EPS_BALL):
        ''' Computes the average of Psi(n_xi + 1)
            p = np.inf for the max-norm
        '''
        tree = ss.cKDTree(points)
        n_points = [np.maximum(len(tree.query_ball_point(point, dist - eps, p=np.inf)), 1) for point, dist in zip(points, dvec)]
        return np.mean(psi(n_points))

    def _psi_avg_chunk(self, points, dvec, M, eps = EPS_BALL):
        ''' Performs the computation on chunks '''
        n_chunks = int(points.shape[0]/M)
        ix = np.random.permutation(points.shape[0])
        points_chunks = np.array_split(points[ix], n_chunks)
        dvec_chunks = np.array_split(dvec[ix], n_chunks)
        avg = 0
        for points_chunk, dvec_chunk in zip(points_chunks, dvec_chunks):
            avg += self._psi_avg(points_chunk, dvec_chunk, eps = eps)
        avg /= n_chunks
        return avg

    def _get_criterion_krashov(self, img1, img2, k = 1, base = 2, chunk = None):
        ''' Estimates the MI with the Krashov formula '''
        x = self._sample(img1)
        y = self._sample(img2)
        
        z = self._joint((x, y))
        dvec = np.maximum(self._nearest_distances(z, k = k), NOISE)
        
        if isinstance(chunk, int):
            return (psi(k) + psi(x.shape[0]) - self._psi_avg_chunk(x, dvec, M=chunk) - self._psi_avg_chunk(y, dvec, M=chunk)) / np.log(base)
        else:
            return (psi(k) + psi(x.shape[0]) - self._psi_avg(x, dvec) - self._psi_avg(y, dvec)) / np.log(base)
        
    def _entropy_KL(self, samples, eps = NOISE):
        ''' Estimate entropy with the K-L formula '''
        N, d = samples.shape
        dvec = self._nearest_distances(samples)

        return d * np.mean(np.log(np.maximum(dvec, eps))) + np.log(((N - 1) * pow(np.pi, d / 2)) / gamma(1 + d / 2)) + EULER_CONSTANT

    def _batch_entropy_KL(self, samples, M=400, eps=NOISE):
        ''' Estimates averaged chunk entropy '''
        n_chunks = int(samples.shape[0] / M)
        chunks = np.array_split(np.random.permutation(samples), n_chunks)
        ent = 0
        for chunk in chunks:
            ent += self._entropy_KL(chunk, eps=eps)

        ent /= n_chunks
        return ent


    def _get_criterion_kybic(self, img1, img2, k = 1, base = 2, chunk = None):
        ''' Estimates mutual information with the Kybic formula '''
        x = self._sample(img1)
        y = self._sample(img2)

        z = self._joint((x, y))
        if isinstance(chunk, int):
            return self._batch_entropy_KL(x, M = chunk) + self._batch_entropy_KL(y, M = chunk) - self._batch_entropy_KL(z, M = chunk)
        else:
            return self._entropy_KL(x) + self._entropy_KL(y) - self._entropy_KL(z)


    def _get_criterion_ite(self, img1, img2, k = 1, base = 2, chunk = None):
        ''' Wrapper around the ITE package, estimates the MI '''
        x = self._sample(img1)
        y = self._sample(img2)
        
        z = self._joint((x, y))

        estimator = MIShannon_HS(k = k)

        return estimator.estimation(z, [x.shape[1], y.shape[1]], chunk = chunk)
        
    def get_criterion(self, img1, img2, k = 1, base = 2, chunk = None, canal = None):
        if self.mode == "kybic":
            return self._get_criterion_kybic(img1, img2, k = k, base = base, chunk = chunk)
        elif self.mode == "krashov":
            return self._get_criterion_krashov(img1, img2, k = k, base = base, chunk = chunk)
        else:
            return self._get_criterion_ite(img1, img2, k = k, base = base, chunk = chunk)


class CoMI(BaseKnnMIEstimator):
    ''' Class for CoMI estimation, inherits from BaseKnnEstimator '''
    def __init__(self, mode):
        super().__init__(mode)
    
    def _sample(self, img):
        ''' Returns the RGB vectors of each pixel, size N_pixel * d (=3) '''
        return np.array([img[i][j] for i in range(img.shape[0]) for j in range(img.shape[1])]) #.astype('float')

    def get_name(self):
        if self.mode == "kybic":
            return "CoMI KYBIC"
        elif self.mode == "krashov":
            return "CoMI KRASHOV"
        else:
            return "CoMI ITE"



class NbMI(BaseKnnMIEstimator):
    def __init__(self, mode):
        super().__init__(mode)

    def _pad_with(self, img, pad_width, iaxis, kwargs):
        ## Utility to pad images, usefull to extract the window samples
        pad_value = kwargs.get('padder', 10)
        img[:pad_width[0]] = pad_value
        img[-pad_width[1]:] = pad_value
        return img

    def _sample(self, img, h = 2):
        ''' Returns the pixel values withing the window, array of size N_pixel * (2h+1)^2 '''
        _img = np.pad(img.astype(int), 2, self._pad_with, padder = -1)
        return np.array([_img[i-h:i+h+1][:,j-h:j+h+1].flatten() for i in np.arange(h, img.shape[0]+h) for j in np.arange(h, img.shape[1]+h)]) #.astype("float")

    def get_name(self):
        if self.mode == "kybic":
            return "NbMI KYBIC"
        elif self.mode == "krashov":
            return "NbMI KRASHOV"
        else:
            return "NbMI ITE"