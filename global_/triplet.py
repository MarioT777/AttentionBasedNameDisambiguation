'''
Author: your name
Date: 2021-03-17 22:37:38
LastEditTime: 2021-03-22 21:18:11
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /AttentionBasedNameDisambiguation/global_/triplet.py
'''
from keras import backend as K


def l2Norm(x):
    return K.l2_normalize(x, axis=-1)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def triplet_loss(_, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))


def accuracy(_, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])


