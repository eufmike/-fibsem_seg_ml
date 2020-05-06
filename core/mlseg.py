import os, sys
import itertools
import numpy as np

# image
from imutils import paths
import cv2
from skimage.io import imread, imsave, imshow
from skimage import util, feature

from PIL import Image, ImageTk

# figure
import matplotlib.pyplot as plt

# metadata
import uuid
import json
from pprint import pprint
from datetime import datetime

from numpy.lib import stride_tricks
import mahotas as mt
import time
import pickle as pkl
import progressbar
import argparse

from sklearn.model_selection import train_test_split
from sklearn import metrics

from tqdm.notebook import trange, tqdm
from tqdm import tnrange, tqdm_notebook

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# customized function
from core.imageprep import dir_checker, random_crop, crop_generator, random_crop_batch

def create_binary_pattern(img, p, r):
    # print ('[INFO] Computing local binary pattern features.')
    lbp = feature.local_binary_pattern(img, p, r)
    return (lbp-np.min(lbp))/(np.max(lbp)-np.min(lbp)) * 255

def create_features(img, img_gray, label, sampling=True, lbp_radius = 24, h_neigh = 11, num_examples = 1000):
    
    # set parameters
    # lbp_radius: local binary pattern neighbourhood, default = 24
    # h_neigh: haralick neighbourhood, default = 11
    # num_examples: number of examples per image to use for training model, default = 1000
    
    # create feature image from img
    feature_img = np.zeros((img.shape[0],img.shape[1], 2))
    feature_img[:,:, 0] = img
    img = None

    # add feature: binary pattern
    lbp_points = lbp_radius*8
    feature_img[:,:,1] = create_binary_pattern(img_gray, lbp_points, lbp_radius)
    
    # add feature: Gaussian
    
    # offset the border
    h_ind = int((h_neigh - 1)/ 2)
    feature_img = feature_img[h_ind:-h_ind, h_ind:-h_ind]
    label = label[h_ind:-h_ind, h_ind:-h_ind]
    
    # reshape array to 1-d
    features = feature_img.reshape(feature_img.shape[0]*feature_img.shape[1], feature_img.shape[2])
    labels = label.reshape(label.shape[0]*label.shape[1], 1)
    
    # Select partial pixles when train = True
    # apply to images
    if sampling == True:
        ss_idx = subsample_idx(0, features.shape[0], num_examples)
        features = features[ss_idx]
        labels = labels[ss_idx]
        
    else:
        ss_idx = []

    # add feature: haralick texture feature
    h_features = haralick_features(img_gray, h_neigh, ss_idx) # h_neigh is given
    
    # add feature: haralick texture feature
    # new function: whole img
    '''
    d = (1, 2)
    theta = (0, np.pi/4, np.pi/2, 3*np.pi/4)
    levels = 256
    win = 19
    
    
    h_features_img = haralick_features_s(img = img_gray, 
                                         win = win, 
                                         d = d, 
                                         theta = theta, 
                                         levels = levels, 
                                         props = ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'))
    
    h_features_img = haralick_features_ssm(img = img_gray, win = win, d=d)
    h_features_img = h_features_img[h_ind:-h_ind, h_ind:-h_ind]
    h_features = h_features_img.reshape(feature_img.shape[0]*feature_img.shape[1], feature_img.shape[2])
    h_features = h_features[ss_idx]
    '''

    features = np.hstack((features, h_features))

    return features, labels

def create_features_predict(img, img_gray, lbp_radius = 24, h_neigh = 11, num_examples = 1000):

    # lbp_radius: local binary pattern neighbourhood, default = 24
    # h_neigh: haralick neighbourhood, default = 11
    # num_examples: number of examples per image to use for training model, default = 1000

    lbp_points = lbp_radius*8
    h_ind = int((h_neigh - 1)/ 2)

    feature_img = np.zeros((img.shape[0],img.shape[1], 2))
    feature_img[:,:, 0] = img
    img = None
    feature_img[:,:, 1] = create_binary_pattern(img_gray, lbp_points, lbp_radius)
    print(feature_img.shape)
    feature_img = feature_img[h_ind:-h_ind, h_ind:-h_ind]
    print(feature_img.shape)
    features = feature_img.reshape(feature_img.shape[0]*feature_img.shape[1], feature_img.shape[2])

    ss_idx = []
    h_features = haralick_features(img_gray, h_neigh, ss_idx)
    features = np.hstack((features, h_features))
    
    return features

def create_training_dataset(img_stack, label_stack, num_examples = 1000):
    
    print('Creating training dataset')
    print('Amount of images: {}'.format(img_stack.shape[2]))

    X = []
    y = []

    for i in tqdm(range(img_stack.shape[2]), desc='01_Image#'):
        img = img_stack[:, :, i]
        features, labels = create_features(img, img, label_stack[:, :, i], sampling=True, num_examples = num_examples)
        X.append(features)
        y.append(labels)
    
    return X, y

def create_predict_dataset(img_stack):
    
    print('Creating predict dataset')
    print('Amount of images: {}'.format(img_stack.shape[2]))

    X = []

    for i in trange(img_stack.shape[2]):
        img = img_stack[:, :, i]
        features = create_features_predict(img, img)
        print(features.shape)
        X.append(features)

    X = np.array(X)
    X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])

    return X


def subsample(features, labels, low, high, sample_size):
    idx = np.random.randint(low, high, sample_size)
    return features[idx], labels[idx]

def subsample_idx(low, high, sample_size):
    return np.random.randint(low, high, sample_size)


def calc_haralick(roi, distance=1):
    # perform haralick calculation for each pixel
    feature_vec = []
    texture_features = mt.features.haralick(roi, distance=distance)
    mean_ht = texture_features.mean(axis=0)
    [feature_vec.append(i) for i in mean_ht[0:9]]
    
    return np.array(feature_vec)

def haralick_features(img, h_neigh, ss_idx):
    # print ('Computing haralick features.')
    size = h_neigh
    shape = (img.shape[0] - size + 1, img.shape[1] - size + 1, size, size)
    strides = 2 * img.strides
    patches = stride_tricks.as_strided(img, shape=shape, strides=strides)
    patches = patches.reshape(-1, size, size)

    # trun off the progressbar and use tqdm in stead
    
    h_features = []

    if len(ss_idx) == 0:
        for i in tqdm(range(len(patches)), desc = '02_haralick', leave = False):
            h_features.append(calc_haralick(patches[i]))
    else:
        for i in tqdm(range(len(patches[ss_idx])), desc = '02_haralick', leave = False):
            h_features.append(calc_haralick(patches[i]))

    #h_features = [calc_haralick(p) for p in patches[ss_idx]]

    return np.array(h_features)


# ====== haralick_features from stackoverflow ======
# https://stackoverflow.com/questions/42459493/sliding-window-in-python-for-glcm-calculation


from skimage import io
from scipy import stats
from skimage.feature import greycoprops

def offset(length, angle):
    """Return the offset in pixels for a given length and angle"""
    dv = length * np.sign(-np.sin(angle)).astype(np.int32)
    dh = length * np.sign(np.cos(angle)).astype(np.int32)
    return dv, dh

def crop(img, center, win):
    """Return a square crop of img centered at center (side = 2*win + 1)"""
    row, col = center
    side = 2*win + 1
    first_row = row - win
    first_col = col - win
    last_row = first_row + side    
    last_col = first_col + side
    return img[first_row: last_row, first_col: last_col]

def cooc_maps(img, center, win, d=[1], theta=[0], levels=256):
    """
    Return a set of co-occurrence maps for different d and theta in a square 
    crop centered at center (side = 2*w + 1)
    """
    shape = (2*win + 1, 2*win + 1, len(d), len(theta))
    cooc = np.zeros(shape=shape, dtype=np.int32)
    row, col = center
    Ii = crop(img, (row, col), win)
    for d_index, length in enumerate(d):
        for a_index, angle in enumerate(theta):
            dv, dh = offset(length, angle)
            Ij = crop(img, center=(row + dv, col + dh), win=win)
            cooc[:, :, d_index, a_index] = encode_cooccurrence(Ii, Ij, levels)
    return cooc

def encode_cooccurrence(x, y, levels=256):
    """Return the code corresponding to co-occurrence of intensities x and y"""
    return x*levels + y

def decode_cooccurrence(code, levels=256):
    """Return the intensities x, y corresponding to code"""
    return code//levels, np.mod(code, levels)    

def compute_glcms(cooccurrence_maps, levels=256):
    """Compute the cooccurrence frequencies of the cooccurrence maps"""
    Nr, Na = cooccurrence_maps.shape[2:]
    glcms = np.zeros(shape=(levels, levels, Nr, Na), dtype=np.float64)
    for r in range(Nr):
        for a in range(Na):
            table = stats.itemfreq(cooccurrence_maps[:, :, r, a])
            codes = table[:, 0]
            freqs = table[:, 1]/float(table[:, 1].sum())
            i, j = decode_cooccurrence(codes, levels=levels)
            glcms[i, j, r, a] = freqs
    return glcms

def compute_props(glcms, props=('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM')):
    """Return a feature vector corresponding to a set of GLCM"""
    Nr, Na = glcms.shape[2:]
    features = np.zeros(shape=(Nr, Na, len(props)))
    for index, prop_name in enumerate(props):
        features[:, :, index] = greycoprops(glcms, prop_name)
    return features.ravel()
    

def haralick_features_s(img, win, d, theta, levels, props):
    """Return a map of Haralick features (one feature vector per pixel)"""
    rows, cols = img.shape
    margin = win + max(d)
    arr = np.pad(img, margin, mode='reflect')
    n_features = len(d) * len(theta) * len(props)
    feature_map = np.zeros(shape=(rows, cols, n_features), dtype=np.float64)
    
    # for m in range(rows):
    for m in tqdm(range(rows), desc='02_rows', leave=False):
        for n in tqdm(range(cols), desc='03_cols', leave=False):
        # for n in range(cols):
            coocs = cooc_maps(arr, (m + margin, n + margin), win, d, theta, levels)
            glcms = compute_glcms(coocs, levels)
            feature_map[m, n, :] = compute_props(glcms, props)
    return feature_map

   
##  Third options
def haralick_features_ss(img, win, d):
    win_sz = 2*win + 1
    window_shape = (win_sz, win_sz)
    arr = np.pad(img, win, mode='reflect')
    windows = util.view_as_windows(arr, window_shape)
    Nd = len(d)
    feats = np.zeros(shape=windows.shape[:2] + (Nd, 4, 13), dtype=np.float64)
    for m in tqdm(range(windows.shape[0]), desc='02_rows', leave=False):
        for n in tqdm(range(windows.shape[1]), desc='03_cols', leave=False):
            for i, di in enumerate(d):
                w = windows[m, n, :, :]
                feats[m, n, i, :, :] = mt.features.haralick(w, distance=di)
    return feats.reshape(feats.shape[:2] + (-1,))


##  Third options mean
def haralick_features_ssm(img, win, d):
    win_sz = 2*win + 1
    window_shape = (win_sz, win_sz)
    arr = np.pad(img, win, mode='reflect')
    windows = util.view_as_windows(arr, window_shape)
    Nd = len(d)
    feats = np.zeros(shape=windows.shape[:2] + (Nd, 9), dtype=np.float64)
    for m in tqdm_notebook(range(windows.shape[0]), desc='02_rows', leave=False):
        for n in tqdm_notebook(range(windows.shape[1]), desc='03_cols', leave=False):
            for i, di in enumerate(d):
                w = windows[m, n, :, :]
                feats[m, n, i, :] = calc_haralick(w, distance=di)
    return feats.reshape(feats.shape[:2] + (-1,))


















##  Third options parallel

import multiprocessing
from functools import partial

data_list = [1, 2, 3, 4]

def prod_xy(x,y):
    return x * y

def parallel_runs(data_list):
    pool = multiprocessing.Pool(processes=4)
    prod_x=partial(prod_xy, y=10) # prod_x has only one argument x (y is fixed to 10)
    result_list = pool.map(prod_x, data_list)
    print(result_list)

if __name__ == '__main__':
    parallel_runs(data_list)

def n_processing(img, n):
    w = windows[m, n, :, :]
    feats[m, n, i, :, :] = mt.features.haralick(w, distance=di)
    return
    
def haralick_features_ssp(img, win, d):
    win_sz = 2*win + 1
    window_shape = (win_sz, win_sz)
    arr = np.pad(img, win, mode='reflect')
    windows = util.view_as_windows(arr, window_shape)
    Nd = len(d)
    feats = np.zeros(shape=windows.shape[:2] + (Nd, 4, 13), dtype=np.float64)
    for m in tqdm(range(windows.shape[0]), desc='02_rows', leave=False):
        for n in tqdm(range(windows.shape[1]), desc='03_cols', leave=False):
            for i, di in enumerate(d):
                w = windows[m, n, :, :]
                feats[m, n, i, :, :] = mt.features.haralick(w, distance=di)
    return feats.reshape(feats.shape[:2] + (-1,))

