'''
Author: your name
Date: 2021-05-10 11:29:34
LastEditTime: 2021-05-10 17:28:56
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /AttentionBasedNameDisambiguation/Adaltion/TripletLossAdaltion.py
'''

from os.path import join
import os
import numpy as np
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam
from global_.triplet import l2Norm, euclidean_distance, triplet_loss, accuracy
from global_.embedding import EMB_DIM
from utils import eval_utils
from utils import data_utils
from utils import settings
from sklearn.metrics import roc_auc_score


"""
global metric learning model
"""


class GlobalTripletModel:

    def __init__(self, data_scale):

        self.data_scale = data_scale
        self.train_triplets_dir = join(settings.OUT_DIR, 'triplets-{}'.format(self.data_scale))
        self.test_triplets_dir = join(settings.OUT_DIR, 'test-triplets')
        self.train_triplet_files_num = self.get_triplets_files_num(self.train_triplets_dir)
        self.test_triplet_files_num = self.get_triplets_files_num(self.test_triplets_dir)
        print('test file num', self.test_triplet_files_num)

    @staticmethod
    def get_triplets_files_num(path_dir):
        files = []
        for f in os.listdir(path_dir):
            if f.startswith('anchor_embs_'):
                files.append(f)
        return len(files)

    def load_batch_triplets(self, f_idx, role='train'):
        if role == 'train':
            cur_dir = self.train_triplets_dir
        else:
            cur_dir = self.test_triplets_dir
        X1 = data_utils.load_data(cur_dir, 'anchor_embs_{}_{}.pkl'.format(role, f_idx))
        X2 = data_utils.load_data(cur_dir, 'pos_embs_{}_{}.pkl'.format(role, f_idx))
        X3 = data_utils.load_data(cur_dir, 'neg_embs_{}_{}.pkl'.format(role, f_idx))
        return X1, X2, X3

    def load_triplets_data(self, role='train'):
        X1 = np.empty([0, EMB_DIM])
        X2 = np.empty([0, EMB_DIM])
        X3 = np.empty([0, EMB_DIM])
        if role == 'train':
            f_num = self.train_triplet_files_num
        else:
            f_num = self.test_triplet_files_num
        for i in range(f_num):
            print('load', i)
            x1_batch, x2_batch, x3_batch = self.load_batch_triplets(i, role)
            p = np.random.permutation(len(x1_batch))
            x1_batch = np.array(x1_batch)[p]
            x2_batch = np.array(x2_batch)[p]
            x3_batch = np.array(x3_batch)[p]
            X1 = np.concatenate((X1, x1_batch))
            X2 = np.concatenate((X2, x2_batch))
            X3 = np.concatenate((X3, x3_batch))
        return X1, X2, X3

    @staticmethod
    def create_triplet_model():
        emb_anchor = Input(shape=(EMB_DIM, ), name='anchor_input')
        emb_pos = Input(shape=(EMB_DIM, ), name='pos_input')
        emb_neg = Input(shape=(EMB_DIM, ), name='neg_input')

        # shared layers
        layer1 = Dense(128, activation='relu', name='first_emb_layer')
        layer2 = Dense(64, activation='relu', name='last_emb_layer')
        norm_layer = Lambda(l2Norm, name='norm_layer', output_shape=[64])

        encoded_emb = norm_layer(layer2(layer1(emb_anchor)))
        encoded_emb_pos = norm_layer(layer2(layer1(emb_pos)))
        encoded_emb_neg = norm_layer(layer2(layer1(emb_neg)))

        pos_dist = Lambda(euclidean_distance, name='pos_dist')([encoded_emb, encoded_emb_pos])
        neg_dist = Lambda(euclidean_distance, name='neg_dist')([encoded_emb, encoded_emb_neg])

        def cal_output_shape(input_shape):
            shape = list(input_shape[0])
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)

        stacked_dists = Lambda(
            lambda vects: K.stack(vects, axis=1),
            name='stacked_dists',
            output_shape=cal_output_shape
        )([pos_dist, neg_dist])

        model = Model([emb_anchor, emb_pos, emb_neg], stacked_dists, name='triple_siamese')
        model.compile(loss=triplet_loss, optimizer=Adam(lr=0.01), metrics=[accuracy])

        inter_layer = Model(inputs=model.get_input_at(0), outputs=model.get_layer('norm_layer').get_output_at(0))

        return model, inter_layer

    # 模型输出路径要改，改成Adaltion TripletLossAdaltion
    def load_triplets_model(self):
        model_dir = join(settings.OUT_DIR, 'model')
        rf = open(join(model_dir, 'adaltion_tripletloss_model-triplets-{}.json'.format(self.data_scale)), 'r')
        model_json = rf.read()
        rf.close()
        loaded_model = model_from_json(model_json)
        loaded_model.load_weights(join(model_dir, 'adaltion_tripletloss_model-triplets-{}.h5'.format(self.data_scale)))
        return loaded_model

    def train_triplets_model(self):
        X1, X2, X3 = self.load_triplets_data()
        n_triplets = len(X1)
        print('loaded')
        model, inter_model = self.create_triplet_model()
        # print(model.summary())

        X_anchor, X_pos, X_neg = X1, X2, X3
        X = {'anchor_input': X_anchor, 'pos_input': X_pos, 'neg_input': X_neg}
        model.fit(X, np.ones((n_triplets, 2)), batch_size=64, epochs=5, shuffle=True, validation_split=0.2)

        model_json = model.to_json()


        model_dir = join(settings.OUT_DIR, 'model')


        os.makedirs(model_dir, exist_ok=True)
        with open(join(model_dir, 'adaltion_tripletloss_model-triplets-{}.json'.format(self.data_scale)), 'w') as wf:
            wf.write(model_json)
        model.save_weights(join(model_dir, 'adaltion_tripletloss_model-triplets-{}.h5'.format(self.data_scale)))

        test_triplets = self.load_triplets_data(role='test')
        auc_score = self.full_auc(model, test_triplets)
        # print('AUC', auc_score)

        loaded_model = self.load_triplets_model()
        print('triplets model loaded')

        auc_score = self.full_auc(loaded_model, test_triplets)


    def evaluate_triplet_model(self):
        test_triplets = self.load_triplets_data(role='test')
        loaded_model = self.load_triplets_model()
        print('triplets model loaded')

        auc_score = self.full_auc(loaded_model, test_triplets)

    
    def get_hidden_output(self, model, inp):
        get_activations = K.function(model.inputs[:1] + [K.learning_phase()], [model.get_layer('norm_layer').get_output_at(0), ])
        activations = get_activations([inp, 0])
        return activations[0]


    def full_auc(self, model, test_triplets):
        grnds = []
        preds = []
        preds_before = []
        embs_anchor, embs_pos, embs_neg = test_triplets

        inter_embs_anchor = self.get_hidden_output(model, embs_anchor)
        inter_embs_pos = self.get_hidden_output(model,embs_pos)
        inter_embs_neg = self.get_hidden_output(model, embs_neg)

        print("===== inter_embs_anchor =====")
        print(embs_anchor)
        print(inter_embs_anchor)
        print("===== inter_embs_anchor =====")


        accs = []
        accs_before = []

        for i, e in enumerate(inter_embs_anchor):
            if i % 10000 == 0:
                print('test', i)

            emb_anchor = e
            emb_pos = inter_embs_pos[i]
            emb_neg = inter_embs_neg[i]
            test_embs = np.array([emb_pos, emb_neg])

            emb_anchor_before = embs_anchor[i]
            emb_pos_before = embs_pos[i]
            emb_neg_before = embs_neg[i]
            test_embs_before = np.array([emb_pos_before, emb_neg_before])

            predictions = eval_utils.predict(emb_anchor, test_embs)
            predictions_before = eval_utils.predict(emb_anchor_before, test_embs_before)

            acc_before = 1 if predictions_before[0] < predictions_before[1] else 0
            acc = 1 if predictions[0] < predictions[1] else 0
            accs_before.append(acc_before)
            accs.append(acc)

            grnd = [0, 1]
            grnds += grnd
            preds += predictions
            preds_before += predictions_before

        # print("======= check grnds and preds_before =======")

        # print(grnds)
        # print(preds_before)

        # print("======= check grnds and preds_before =======")


        auc_before = roc_auc_score(grnds, preds_before)
        auc = roc_auc_score(grnds, preds)
        print('test accuracy before', np.mean(accs_before))
        print('test accuracy after', np.mean(accs))

        print('test AUC before', auc_before)
        print('test AUC after', auc)
        return auc




# 所有目录都要改

if __name__ == '__main__':
    global_model = GlobalTripletModel(data_scale=1000000)
    global_model.train_triplets_model()
    print('done')
    
