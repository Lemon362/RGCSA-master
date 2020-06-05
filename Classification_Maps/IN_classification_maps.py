# -*- coding: utf-8 -*-
# 描述分类结果，主要是输出色彩图，这个过程比较麻烦的，直接加载模型权重
import codecs
import json

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from tensorflow.python.keras.utils.np_utils import to_categorical
from scipy.io import savemat
from sklearn.decomposition import PCA
from tensorflow.python.keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
import keras.callbacks as kcallbacks
from tensorflow.python.keras.regularizers import l2
import time
import collections
from sklearn import metrics, preprocessing
from tensorflow import Tensor
from tensorflow.python.framework import ops

from Utils import zeroPadding, normalization, doPCA, modelStatsRecord, averageAccuracy, \
    RGCSA, RGCA, RSA
import os
from keras.utils import plot_model
from pylab import *
from tensorflow.python.keras import backend as K

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4


def sampling(proptionVal, groundTruth):  # divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
    # whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        #        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices


def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index


def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len), :]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len)]
    return selected_patch


# 特征图输出
def classification_map(map, groundTruth, dpi, savePath):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1] * 2.0 / dpi, groundTruth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi=dpi)

    return 0


def model_DenseNet():
    model_dense = RGCSA.ResneXt_IN((1, img_rows, img_cols, img_channels),
                                   cardinality=8, classes=16)

    RMS = RMSprop(lr=0.0003)

    # Let's train the model using RMSprop

    def mycrossentropy(y_true, y_pred, e=0.1):
        loss1 = K.categorical_crossentropy(y_true, y_pred)
        loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / nb_classes, y_pred)
        return (1 - e) * loss1 + e * loss2

    model_dense.compile(loss=mycrossentropy, optimizer=RMS, metrics=['accuracy'])

    return model_dense


mat_data = sio.loadmat('D:/RGCSA/Datasets/IN/Indian_pines_corrected.mat')
data_IN = mat_data['indian_pines_corrected']
mat_gt = sio.loadmat('D:/RGCSA/Datasets/IN/Indian_pines_gt.mat')
gt_IN = mat_gt['indian_pines_gt']
print(data_IN.shape)

new_gt_IN = gt_IN

batch_size = 16
nb_classes = 16
nb_epoch = 100  # 400
img_rows, img_cols = 16, 16  # 27, 27
patience = 100

INPUT_DIMENSION_CONV = 200
INPUT_DIMENSION = 200

# 10%:10%:80% data for training, validation and testing

TOTAL_SIZE = 10249
VAL_SIZE = 1025

# 20%:10%:70% data for training, validation and testing

TRAIN_SIZE = 3081
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VALIDATION_SPLIT = 0.7

ALL_SIZE = data_IN.shape[0] * data_IN.shape[1]

img_channels = 200
PATCH_LENGTH = 8  # Patch_size (13*2+1)*(13*2+1)

data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

data = preprocessing.scale(data)

# scaler = preprocessing.MaxAbsScaler()
# data = scaler.fit_transform(data)

data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
whole_data = data_
padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)

ITER = 1
CATEGORY = 16

# 21025,11,11,200
print(ALL_SIZE)
all_data = np.zeros((ALL_SIZE, 2 * PATCH_LENGTH, 2 * PATCH_LENGTH, INPUT_DIMENSION_CONV))
train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH, 2 * PATCH_LENGTH, INPUT_DIMENSION_CONV))
test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH, 2 * PATCH_LENGTH, INPUT_DIMENSION_CONV))

seeds = [1334]

for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))

    best_weights_DenseNet_path = 'D:/RGCSA/models/' \
                                 'Indian_best_RGCSA_3_1_6_100_' + str(
        index_iter + 1) + '.hdf5'

    np.random.seed(index_iter)

    #    train_indices, test_indices = sampleFixNum.samplingFixedNum(TRAIN_NUM, gt)
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)

    y_train_raw = gt[train_indices] - 1
    y_train = to_categorical(np.asarray(y_train_raw))

    y_test_raw = gt[test_indices] - 1
    y_test = to_categorical(np.asarray(y_test_raw))

    all_assign = indexToAssignment(range(ALL_SIZE), whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(all_assign)):
        all_data[i] = selectNeighboringPatch(padded_data, all_assign[i][0], all_assign[i][1], PATCH_LENGTH)

    # first principal component training data
    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    # train_data = np.zeros((len(train_assign), 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    # first principal component testing data
    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    # test_data = np.zeros((len(test_assign), 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    # load trained model
    model_densenetss = model_DenseNet()

    # 加载模型权重
    model_densenetss.load_weights(best_weights_DenseNet_path)

    pred_test_conv1 = model_densenetss.predict(
        all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], all_data.shape[3], 1)).argmax(axis=1)

    print('#' * 100)
    print(pred_test_conv1)
    # [ 2  2  2 ..., 13 13 13]
    print(pred_test_conv1.shape)
    # (21025,)

    x = np.ravel(pred_test_conv1)
    print(x)
    print(x.shape)
    # [ 2  2  2 ..., 13 13 13]
    # (21025,)
    y = np.zeros((x.shape[0], 3))
    print(y)
    print(y.shape)
    # 是把图上每一个像素点都变成了三基色表示
    # [ 0.  0.  0.]
    # (21025, 3)

    # 评估每一类的输出图，并对每一个预测数据进行上色
    for index, item in enumerate(x):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.

    # print y

    y_re = np.reshape(y, (gt_IN.shape[0], gt_IN.shape[1], 3))

    classification_map(y_re, gt_IN, 24,
                       "D:/RGCSA/CMaps/IN_RGCSA_3_1_6_100_1.png")
