# -*- coding: utf-8 -*-
# @Author  : Peida Wu
# @Time    : 20.3.23 023 15:15:47
# @Function:
# -*- coding: utf-8 -*-
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

from Utils import zeroPadding, normalization, doPCA, modelStatsRecord, averageAccuracy, RGCSA, RGCA, RSA
import os
from keras.utils import plot_model
from pylab import *
from tensorflow.python.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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


# divide dataset into train and test datasets
def sampling(proptionVal, groundTruth):
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    print(m)

    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]

        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]

    train_indices = []
    test_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    print(len(test_indices))
    # 8194
    print(len(train_indices))
    # 2055
    return train_indices, test_indices


def model_DenseNet():
    model_dense = RGCSA.ResneXt_IN((1, img_rows, img_cols, img_channels), classes=9)

    RMS = RMSprop(lr=0.0003)

    def mycrossentropy(y_true, y_pred, e=0.1):
        loss1 = K.categorical_crossentropy(y_true, y_pred)

        loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / nb_classes, y_pred)  # K.ones_like(y_pred) / nb_classes

        return (1 - e) * loss1 + e * loss2

    model_dense.compile(loss=mycrossentropy, optimizer=RMS, metrics=['accuracy'])  # categorical_crossentropy

    model_dense.summary()
    # plot_model(model_dense, show_shapes=True, to_file='./model_ResNeXt_GroupChannel_Space_Attention_UP.png')

    return model_dense


mat_data = sio.loadmat('./Datasets/UP/PaviaU.mat')
data_IN = mat_data['paviaU']

mat_gt = sio.loadmat('./Datasets/UP/PaviaU_gt.mat')
gt_IN = mat_gt['paviaU_gt']

print('data_IN shape:', data_IN.shape)
# (145,145,200)
print('gt_IN shape:', gt_IN.shape)
# (145,145)

new_gt_IN = gt_IN

batch_size = 16
nb_classes = 9
nb_epoch = 100

img_rows, img_cols = 16, 16

patience = 100

INPUT_DIMENSION_CONV = 103
INPUT_DIMENSION = 103

TOTAL_SIZE = 42776  # 42776
VAL_SIZE = 4281  # 4281

TRAIN_SIZE = 8558  # 4281 8558 12838 17113 21391 25670
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE  # 38495 34218 29938 25663 21385 17106
# Train:Val:Test=1:1:8 2:1:7 3:1:6 4:1:5 5:1:4 6:1:3
VALIDATION_SPLIT = 0.8  # Val+Test=0.9 0.8 0.7 0.6 0.5

img_channels = 103  # 200
# TODO 和可变参数1一起改变，7--15，3--7
PATCH_LENGTH = 8

print('data_IN.shape[:2]:', data_IN.shape[:2])
# (145,145)
print('np.prod(data_IN.shape[:2]:', np.prod(data_IN.shape[:2]))
# 21025 = 145 * 145
print('data_IN.shape[2:]:', data_IN.shape[2:])
# (200,)
print('np.prod(data_IN.shape[2:]:', np.prod(data_IN.shape[2:]))
# 200
print('np.prod(new_gt_IN.shape[:2]:', np.prod(new_gt_IN.shape[:2]))
# 21025


data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

data = preprocessing.scale(data)
print('data.shape:', data.shape)

data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
whole_data = data_

padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)
print('padded_data.shape:', padded_data.shape)

ITER = 1
CATEGORY = 9

train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH, 2 * PATCH_LENGTH, INPUT_DIMENSION_CONV))
print('train_data.shape:', train_data.shape)
# (2055, 11, 11, 200)
test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH, 2 * PATCH_LENGTH, INPUT_DIMENSION_CONV))
print('test_data.shape:', test_data.shape)
# (8194, 11, 11, 200)


KAPPA_3D_DenseNet = []
OA_3D_DenseNet = []
AA_3D_DenseNet = []
TRAINING_TIME_3D_DenseNet = []
TESTING_TIME_3D_DenseNet = []
ELEMENT_ACC_3D_DenseNet = np.zeros((ITER, CATEGORY))

seeds = [1334]

for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))

    best_weights_DenseNet_path = './models/UP_best_RGCSA_2_1_7_100_' + str(index_iter + 1) + '.hdf5'

    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    # train_indices 5128     test_indices(val+test) 5121

    y_train = gt[train_indices] - 1
    y_train = to_categorical(np.asarray(y_train))

    y_test = gt[test_indices] - 1
    y_test = to_categorical(np.asarray(y_test))

    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION_CONV)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)

    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]

    model_densenet = model_DenseNet()

    earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')

    saveBestModel6 = kcallbacks.ModelCheckpoint(best_weights_DenseNet_path, monitor='val_loss', verbose=1,
                                                save_best_only=True,
                                                mode='auto')

    tic6 = time.clock()
    print(x_train.shape, x_test.shape)
    # (2055,7,7,200)  (7169,7,7,200)
    history_3d_densenet = model_densenet.fit(
        x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1), y_train,
        validation_data=(x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], x_val.shape[3], 1), y_val),
        batch_size=batch_size,
        epochs=nb_epoch, shuffle=True, callbacks=[earlyStopping6, saveBestModel6])
    toc6 = time.clock()

    with open('./Loss_Acc/UP_RGCSA_2_1_7_100_1.hdf5', 'w') as f:
        json.dump(history_3d_densenet.history, f)

    plt.plot(history_3d_densenet.history['acc'])
    plt.plot(history_3d_densenet.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    plt.plot(history_3d_densenet.history['loss'])
    plt.plot(history_3d_densenet.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    tic7 = time.clock()
    loss_and_metrics = model_densenet.evaluate(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1), y_test,
        batch_size=batch_size)
    toc7 = time.clock()

    print('3D DenseNet Time: ', toc6 - tic6)
    print('3D DenseNet Test time:', toc7 - tic7)

    print('3D DenseNet Test score:', loss_and_metrics[0])
    print('3D DenseNet Test accuracy:', loss_and_metrics[1])

    pred_test = model_densenet.predict(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)).argmax(axis=1)

    collections.Counter(pred_test)

    gt_test = gt[test_indices] - 1

    overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
    each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])
    KAPPA_3D_DenseNet.append(kappa)
    OA_3D_DenseNet.append(overall_acc)
    AA_3D_DenseNet.append(average_acc)
    TRAINING_TIME_3D_DenseNet.append(toc6 - tic6)
    TESTING_TIME_3D_DenseNet.append(toc7 - tic7)
    ELEMENT_ACC_3D_DenseNet[index_iter, :] = each_acc

    print("3D DenseNet finished.")
    print("# %d Iteration" % (index_iter + 1))

modelStatsRecord.outputStats(KAPPA_3D_DenseNet, OA_3D_DenseNet, AA_3D_DenseNet, ELEMENT_ACC_3D_DenseNet,
                             TRAINING_TIME_3D_DenseNet, TESTING_TIME_3D_DenseNet,
                             history_3d_densenet, loss_and_metrics, CATEGORY,
                             './records/UP_train_RGCSA_2_1_7_100_1.txt',
                             './records/UP_train_element_RGCSA_2_1_7_100_1.txt')
