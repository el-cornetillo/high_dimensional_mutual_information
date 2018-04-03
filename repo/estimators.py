import numpy as np
from PIL import Image
from scipy.special import psi, gamma

from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings('ignore')

from utils import *

NOISE = 1e-10 ## Noise added to data to avoid degenerancies
EULER_CONSTANT = -psi(1) ## Euler constant, ~0.577


class SSD:
    ''' Class to calculate the negative sum of squares '''
    def __init__(self):
        pass

    def _sample(self, img):
        ''' Returns the RGB vectors of each pixel, size N_pixel * d (=3) '''
        if len(img.shape) == 3:
            return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
        else:
            return img.flatten().reshape(-1 , 1)
    
    def get_criterion(self, img1, img2, chunk =None):
        # params chunk : for consistency, ignored in this function
        
        x = self._sample(img1)
        y = self._sample(img2)

        return -1 * np.sum(np.linalg.norm(x.astype('float') - y.astype('float'), axis = 1)**2)

    def get_name(self):
        return "SSD"

class MI:
    ''' Class to estimate the MI for 2D-images based on the histogram
        method '''
    def __init__(self):
        pass

    def _to_grayscale(self, img):
        return np.asarray(Image.fromarray(img).convert('L'))
    
    def _hist(self, img, img2 = None):
        ''' Computes the histogram (or joint histogram) of the data '''
        if img2 is None:
            return np.histogram(self._to_grayscale(img).flatten())[0] / (img.shape[0] * img.shape[1])

        return np.histogram2d(self._to_grayscale(img).flatten(), self._to_grayscale(img2).flatten())[0].flatten() / (img.shape[0] * img.shape[1])

    def _log(self, x):
        if x <=0.:
            return 0
        return x * np.log(x)

    def _entropy(self, img, img2 = None, base = 2):
        _probs = self._hist(img, img2)
        return (-1 * np.sum([self._log(p) for p in _probs])) / np.log(base)
    
    def get_criterion(self, img1, img2, chunk = None):
        # param chunk : for consistency, ignored in this function
        return self._entropy(img1) + self._entropy(img2) \
                    - self._entropy(img1, img2)

    def get_name(self):
        return "MI"



class BaseKnnMIEstimator:
    ''' Base class for the multi-dimensional MI estimators '''
    def __init__(self):
        pass

    def _joint(self, *kwargs):
        return np.hstack(*kwargs)
    
    def _nearest_distances(self, X, k=1):
        ''' Return the nearest distance from each point to his k-th nearest neighboors '''
        knn = NearestNeighbors(n_neighbors=k+1)
        knn.fit(X)
        d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
        return d[:, -1] # returns the distance to the kth nearest neighbor
        
    def _entropy(self, samples, eps = NOISE):
        ''' Estimate entropy with the K-L formula '''
        N, d = samples.shape
        dvec = self._nearest_distances(samples)

        return d * np.mean(np.log(np.maximum(dvec, eps))) + np.log(((N - 1) * pow(np.pi, d / 2)) / gamma(1 + d / 2)) + EULER_CONSTANT

    def _batch_entropy(self, samples, M=100, eps=NOISE):
        ''' Estimates averaged chunk entropy '''
        N = samples.shape[0]
        h = 0
        cnt = 0
        samples = np.random.permutation(samples)
        for i in np.arange(0, N, M):
            j = i + M
            if j > N:
                continue
            h += self._entropy(samples[i:j], eps=eps)
            cnt += 1

        return h / cnt


    def get_criterion(self, img1, img2, k = 1, base = 2, chunk = None):
        ''' Estimates mutual information with the Kybic formula '''
        x = self._sample(img1)
        y = self._sample(img2)

        z = self._joint((x, y))
        if isinstance(chunk, int):
            return self._batch_entropy(x, M = chunk) + self._batch_entropy(y, M = chunk) - self._batch_entropy(z, M = chunk)
        else:
            return self._entropy(x) + self._entropy(y) - self._entropy(z)


class CoMI(BaseKnnMIEstimator):
    ''' Class for CoMI estimation, inherits from BaseKnnEstimator '''
    def __init__(self):
        super().__init__()
    
    def _sample(self, img):
        ''' Returns the RGB vectors of each pixel, size N_pixel * d (=3) '''
        return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

    def get_name(self):
        return "CoMI"


class NbMI(BaseKnnMIEstimator):
    def __init__(self):
        super().__init__()

    def _to_grayscale(self, img):
        return np.asarray(Image.fromarray(img).convert('L'))

    def _pad_with(self, img, pad_width, iaxis, kwargs):
        ## Utility to pad images, usefull to extract the window samples
        pad_value = kwargs.get('padder', 10)
        img[:pad_width[0]] = pad_value
        img[-pad_width[1]:] = pad_value
        return img

    def _sample(self, img, h = 2):
        ''' Returns the pixel values withing the window, array of size N_pixel * (2h+1)^2 '''
        _img = self._to_grayscale(img)
        #__img = np.pad(_img.astype(int), 2, self._pad_with, padder = -1)
        #return np.array([__img[i-h:i+h+1][:,j-h:j+h+1].flatten() for i in np.arange(h, _img.shape[0]+h) for j in np.arange(h, _img.shape[1]+h)])
        return np.array([_img[i-h:i+h+1][:,j-h:j+h+1].flatten() for i in np.arange(h, _img.shape[0]-h-1) for j in np.arange(h, _img.shape[1]-h-1)])

    def get_name(self):
        return "NbMI"