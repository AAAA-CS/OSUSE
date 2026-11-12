import os
import torch
import numpy as np
import scipy as sp
import scipy.stats
import random
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.spatial.distance import cdist
import torch.nn.functional as F
from operator import truediv
import torch.utils.data as data
from config.config_Houston import *

def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    # 处理数据 key
    if "Houston13" in image_file:
        data_key = "ori_data"
    elif "Houston18" in image_file:
        data_key = "Houston18"
    elif "Dioni" in image_file:
        data_key = "ori_data"
    elif "Loukia" in image_file:
        data_key = "ori_data"
    elif "ZY_YC_data147" in image_file:
        data_key = "Data"
    elif "GF_YC_data" in image_file:
        data_key = "Data"

    # 处理标签 key
    if "Houston13" in label_file:
        if "7gt" in label_file:
            label_key = "map"
    elif "Houston18" in label_file:
        if "gt" in label_file:
            label_key = "Houston18_OS_gt_merged"
    elif "Dioni_gt_out68_modified" in label_file:
        label_key = "map"
    elif "Loukia_gt_out68_modified" in label_file:
        label_key = "map"
    elif "ZY_YC_gt7_modified" in label_file:
        label_key = "DataClass"
    elif "GF_YC_gt_modified" in label_file:
        label_key = "DataClass"

    data_all = image_data[data_key]
    GroundTruth = label_data[label_key]

    print("GroundTruth标签：", np.unique(GroundTruth))

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))

    data_scaler = preprocessing.scale(data.astype(float))
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    return Data_Band_Scaler, GroundTruth


def get_sample_data(Sample_data, Sample_label, HalfWidth, num_per_class):
    print('get_sample_data() run...')
    print('The original sample data shape:', Sample_data.shape)
    nBand = Sample_data.shape[2]

    data = np.pad(Sample_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    label = np.pad(Sample_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    [Row, Column] = np.nonzero(label)
    m = int(np.max(label))
    print(f'num_class : {m}')

    val = {}
    val_indices = []

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        train[i] = indices[:num_per_class]      # 每类num_per_class个样本为训练样本
        val[i] = indices[num_per_class:]        # 除num_per_class个训练样本以外全为测试样本

    for i in range(m):
        train_indices += train[i]
        val_indices += val[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    #val样本划块
    print('the number of val data:', len(val_indices))
    nVAL = len(val_indices)
    val_data = np.zeros([nVAL, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    val_label = np.zeros([nVAL], dtype=np.int64)
    RandPerm = val_indices
    RandPerm = np.array(RandPerm)

    for i in range(nVAL):
        val_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                                  Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
                                                  :],
                                                  (2, 0, 1))
        val_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    val_label = val_label - 1

    #train
    print('the number of processed data:', len(train_indices))
    nTrain = len(train_indices)
    index = np.zeros([nTrain], dtype=np.int64)
    processed_data = np.zeros([nTrain, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTrain], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTrain):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                          Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :],
                                          (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('sample data shape', processed_data.shape)
    print('sample label shape', processed_label.shape)
    print('get_sample_data() end...')
    return processed_data, processed_label  #, val_data, val_label 验证样本（未返回）


def get_unlabel_data(Sample_data, Sample_label, HalfWidth, max_samples=None): # YC250000
    print('get_unlabel_data() run...')
    data = np.pad(Sample_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    nBand = Sample_data.shape[2]

    [Row, Column] = np.nonzero(Sample_label == 0)

    indices = np.arange(len(Row))
    np.random.shuffle(indices)

    if (max_samples is not None) and (len(indices) > max_samples):
        indices = indices[:max_samples]

    n_Unlabel = len(indices)
    print('未标记样本数量：', n_Unlabel)
    unlabel_data = np.zeros([n_Unlabel, nBand, 2*HalfWidth + 1, 2*HalfWidth + 1], dtype=np.float32)

    # 遍历采样点，取 patch
    for i, idx in enumerate(indices):
        r = Row[idx] + HalfWidth  # 注意：加偏移
        c = Column[idx] + HalfWidth
        unlabel_data[i, :, :, :] = np.transpose(
            data[r - HalfWidth:r + HalfWidth + 1,
            c - HalfWidth:c + HalfWidth + 1, :],
            (2, 0, 1)
        )
    print('get_unlabel_sample_data() end...')
    return unlabel_data


def get_all_data(All_data, All_label, HalfWidth):
    print('get_all_data() run...')
    print('The original data shape:', All_data.shape)
    nBand = All_data.shape[2]

    data = np.pad(All_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    label = np.pad(All_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    [Row, Column] = np.nonzero(label)
    num_class = int(np.max(label))
    print(f'num_class : {num_class}')

    for i in range(num_class):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if
                   label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        train[i] = indices

    for i in range(num_class):
        train_indices += train[i]
    np.random.shuffle(train_indices)

    print('the number of all data:', len(train_indices))
    nTest = len(train_indices)
    index = np.zeros([nTest], dtype=np.int64)
    processed_data = np.zeros([nTest, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTest], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTest):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                          Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :],
                                          (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('processed all data shape:', processed_data.shape)
    print('processed all label shape:', processed_label.shape)
    print('get_all_data() end...')
    return index, processed_data, processed_label, label, RandPerm, Row, Column

def get_coords_from_RandPerm(RandPerm, Row, Column):
    coords = np.stack([Row[RandPerm], Column[RandPerm]], axis=1)
    coords = coords / np.max(coords, axis=0)  # 归一化
    return coords

# set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

