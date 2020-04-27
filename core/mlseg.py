import os, sys
import itertools
import numpy as np

# image
from imutils import paths
import cv2
from skimage.io import imread, imsave, imshow
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

from tqdm.notebook import trange

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from skimage import feature

# customized function
from core.imageprep import dir_checker, random_crop, crop_generator, random_crop_batch

def create_binary_pattern(img, p, r):
    # print ('[INFO] Computing local binary pattern features.')
    lbp = feature.local_binary_pattern(img, p, r)
    return (lbp-np.min(lbp))/(np.max(lbp)-np.min(lbp)) * 255

def create_features(img, img_gray, label, train=True):

    lbp_radius = 24 # local binary pattern neighbourhood
    h_neigh = 11 # haralick neighbourhood
    num_examples = 1000 # number of examples per image to use for training model

    lbp_points = lbp_radius*8
    h_ind = int((h_neigh - 1)/ 2)

    feature_img = np.zeros((img.shape[0],img.shape[1], 2))
    feature_img[:,:, 0] = img
    img = None
    feature_img[:,:,1] = create_binary_pattern(img_gray, lbp_points, lbp_radius)
    feature_img = feature_img[h_ind:-h_ind, h_ind:-h_ind]
    features = feature_img.reshape(feature_img.shape[0]*feature_img.shape[1], feature_img.shape[2])

    if train == True:
        ss_idx = subsample_idx(0, features.shape[0], num_examples)
        features = features[ss_idx]
    else:
        ss_idx = []

    if train == True:
        label = label[h_ind:-h_ind, h_ind:-h_ind]
        labels = label.reshape(label.shape[0]*label.shape[1], 1)
        labels = labels[ss_idx]
    else:
        labels = None

    return features, labels

def create_features_predict(img, img_gray):

    lbp_radius = 24 # local binary pattern neighbourhood
    h_neigh = 11 # haralick neighbourhood
    num_examples = 1000 # number of examples per image to use for training model

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

    return features

def create_training_dataset(img_stack, label_stack):
    
    print('Creating training dataset')
    print('Amount of images: {}'.format(img_stack.shape[2]))

    X = []
    y = []

    for i in trange(img_stack.shape[2]):
        img = img_stack[:, :, i]
        features, labels = create_features(img, img, label_stack[:, :, i], train=True)
        X.append(features)
        y.append(labels)

    X = np.array(X)
    X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
    y = np.array(y)
    y = y.reshape(y.shape[0]*y.shape[1], y.shape[2]).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print ('Feature vector size: {}'.format(X_train.shape))

    return X_train, X_test, y_train, y_test

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


def calc_haralick(roi):
    feature_vec = []
    texture_features = mt.features.haralick(roi)
    mean_ht = texture_features.mean(axis=0)
    [feature_vec.append(i) for i in mean_ht[0:9]]
    return np.array(feature_vec)

def harlick_features(img, h_neigh, ss_idx):

    print ('Computing haralick features.')
    size = h_neigh
    shape = (img.shape[0] - size + 1, img.shape[1] - size + 1, size, size)
    strides = 2 * img.strides
    patches = stride_tricks.as_strided(img, shape=shape, strides=strides)
    patches = patches.reshape(-1, size, size)

    if len(ss_idx) == 0 :
        bar = progressbar.ProgressBar(maxval=len(patches), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    else:
        bar = progressbar.ProgressBar(maxval=len(ss_idx), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    bar.start()

    h_features = []

    if len(ss_idx) == 0:
        for i, p in enumerate(patches):
            bar.update(i+1)
            h_features.append(calc_haralick(p))
    else:
        for i, p in enumerate(patches[ss_idx]):
            bar.update(i+1)
            h_features.append(calc_haralick(p))

    #h_features = [calc_haralick(p) for p in patches[ss_idx]]

    return np.array(h_features)