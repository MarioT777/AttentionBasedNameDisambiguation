'''
Author: your name
Date: 2021-03-17 22:37:38
LastEditTime: 2021-07-07 13:41:52
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /AttentionBasedNameDisambiguation/global_/triplet.py
'''
from keras import backend as K
import tensorflow as tf 

def l2Norm(x):
    return K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon())) 

# def disambiguate_distance(arguments):
#     x, y , center = arguments
#     D_i = euclidean_distance([x, center])
#     D_j = euclidean_distance([y, center])
#     # D_ij = K.math.divide(D_i, D_j, name= 'DisambiguateDivide' )
#     D_ij = D_i / D_j
#     # check the error of 'Tensor' object has no attribute '_keras_history'
#     return D_ij

def disambiguate_distance_rate(arguments):

    # 内嵌函数，因为报了undefine的错
    def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon())) 

    encoded_emb, encoded_emb_pos, emb_atten_pos, emb_atten, emb_center = arguments
    dist = euclidean_distance([encoded_emb, encoded_emb_pos])
    D_i = euclidean_distance([emb_atten_pos, emb_center])
    D_j = euclidean_distance([emb_atten, emb_center])
    # 这里可能会出问题
    D_ij = D_i / D_j
    
    dist = D_ij * dist
    return dist
    
def triplet_loss(_, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def disambiguate_loss(_, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0),  K.square(y_pred[:,0,0]) -  K.square(y_pred[:,1,0]) + margin))


def accuracy(_, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])










