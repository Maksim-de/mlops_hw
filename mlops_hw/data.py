import numpy as np
import pandas as pd
import collections
import gc
from tqdm import tqdm
import cv2
import cudf, cuml, cupy
from cuml.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import transformers
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig


import sys
sys.path.append('../input/timm-shpee/pytorch-image-models-master')
import timm
from timm.models.layers import SelectAdaptivePool2d

cos_threshs = np.array([0.16])
test_batch_size = 128

# image model
GEM_P = 4
image_size = 420

# TTA for image(do not use)
flip_TTAs = [False, False, False, False, False]
testing_scales = [[1.0], [1.0], [1.0], [1.0], [1.0]]

# text model
text_max_length = 84

# alpha query expansion
alpha_query_expansion = True
qe_mid_knn = True
qe_ms     = [[1, 1], [1, 1], [1, 1], [2, 1], [1, 1]]
qe_alphas = [[2, 5], [2, 7], [5, 2], [7, 2], [3, 3]]

# adaptive thresholding
USE_ADAPTIVE_THRESHOLDING = False
CONSERVATIVENESS = 1.0
BETA = np.mean([0.9, 0.8, 0.9, 0.75, 0.3])

# min num preds
force_2preds = True
force_2preds_relax = 1.2

# kNN
KNN = 52
ALPHA_QE_KNN = 8
knn_metric = 'cosine' # cosine or correlation


if COMPUTE_CV:
    test = pd.read_csv('../input/shopee-product-matching/train.csv').iloc[0:300]
else:
    test = pd.read_csv('../input/shopee-product-matching/test.csv')
test = test.drop(columns='image_phash')

LEN_TEST = len(test)

BASE = '../input/shopee-product-matching/test_images/'
if COMPUTE_CV:
    BASE = '../input/shopee-product-matching/train_images/'
    
CHUNK = 1024*4
CTS = LEN_TEST//CHUNK
if LEN_TEST%CHUNK!=0:
    CTS += 1
    
if LEN_TEST==3:
    KNN = 3
    ALPHA_QE_KNN = 3
    qe_ms     = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    qe_alphas = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]