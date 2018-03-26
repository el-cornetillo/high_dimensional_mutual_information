import cv2

from skimage import util
from skimage.transform import resize
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.transform import rotate
from skimage.color import rgb2gray

import numpy as np
import time
import math

from scipy import ndimage
from skimage import feature

import matplotlib.pyplot as plt

from tqdm import tqdm

def select_canal(img, canal = None):
    ''' Convert an 3-dimensional RGB image into a 2D-version '''
    if canal == "intensity":
        return np.mean(img, axis = 2).astype(img.dtype)
    if canal == 'grayscale':
        return (rgb2gray(img)*255.).astype(img.dtype)
    if canal == 'R':
        return img[:, :, 0]
    elif canal == "G":
        return img[:, :, 1]
    elif canal == "B":
        return img[:, :, 2]
    else:
        return img


def largest_rotated_rect(w, h, angle):
    '''Finds the largest rectangle in a rotated rectangle '''
    quadrant = int(np.floor(angle / (np.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else np.pi - angle
    alpha = (sign_alpha % np.pi + np.pi) % np.pi

    bb_w = w * np.cos(alpha) + h * np.sin(alpha)
    bb_h = w * np.sin(alpha) + h * np.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = np.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * np.cos(alpha)
    a = d * np.sin(alpha) / np.sin(delta)

    y = a * np.cos(gamma)
    x = y * np.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    '''Crops an image around its center'''
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


# def rotation(img, angle):

#     image_height, image_width = img.shape[0:2]

#     return np.rint(resize(crop_around_center(
#                 rotate(img.astype('float'), angle=angle, resize=True),
#                 *largest_rotated_rect(
#                     image_width,
#                     image_height,
#                     np.radians(angle)
#                 )
#             ), (image_height, image_width)) * 255.).astype('float')


def rotation(img, angle):
    '''Rotate an image and crops the background so that
       the rotated image has the same shape
    '''
    if angle == 0:
        return img

    image_height, image_width = img.shape[0:2]

    return resize(crop_around_center(
                rotate(img, angle=angle, resize=True),
                *largest_rotated_rect(
                    image_width,
                    image_height,
                    np.radians(angle)
                )
            ), (image_height, image_width))
            
            
def increase_HSV(img, value_h = 30, value_s = 30, value_v = 30):
    ''' Increase H, S, V values (hue, saturation, brightness) '''
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value_v
    v[v > lim] = 255
    v[v <= lim] += value_v
    
    lim = 255 - value_s
    s[s > lim] = 255
    s[s <= lim] += value_s
    
    lim = 255 - value_h
    h[h > lim] = 255
    h[h <= lim] += value_h

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img


def modify(img, hue, saturation, value, mean_noise = 0, var_noise = 0.01):
    ''' Increase H, S, V values, rotates the color map and add some random gaussian noise '''
    return (util.random_noise(cv2.cvtColor(increase_HSV(img, value_h = hue, value_s = saturation, value_v = value), cv2.COLOR_RGB2BGR), \
                mode='gaussian', seed=None, clip=True, var = var_noise, mean=mean_noise) * 255)


def edge_detector(img, sigma = 3):
    ''' Detect edges '''
    return (1 - feature.canny(img, sigma = sigma)).astype(int) * 255

def gaussian_noise(img, sigma = 5):
    ''' Add gaussian noise '''
    return ndimage.gaussian_filter(img, sigma = sigma)


def test_estimator(estimator, img1, img2, chunk = None, canal = None, verbose = False):
    ''' Utility to test a Mutual Information estimator
        Rotates img2 between -10 and 10 degrees and
        calculates the MI.
    '''
    criterions = []
    tic = time.time()
    for angle in tqdm(np.arange(-10, 11, 1)):
        criterion = estimator.get_criterion(uint_scale(img1), uint_scale(rotation(uint_scale(img2), angle=angle)), chunk = chunk, canal = canal) 
        if verbose:
            print('Angle : {} CRITERION {}'.format(angle, criterion))
        criterions.append(criterion)
    toc = time.time()
    if verbose:
        plt.plot(np.arange(-10, 11, 1), min_max_scaler(criterions))
        plt.show()
    
    return criterions, (toc-tic) / 21

def uint_scale(img):
    if np.max(img) <= 1:
        return np.rint(img * 255 / np.max(img)).astype('uint8')
    else:
        return np.rint(img).astype("uint8")

def min_max_scaler(x):
    ''' Utility to linearly scale data between 0 and 1 '''
    m = np.min(x)
    M = np.max(x)
    
    return [(e-m) / (M - m) for e in x]