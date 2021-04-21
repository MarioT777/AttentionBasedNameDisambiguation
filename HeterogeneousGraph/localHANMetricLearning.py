'''
Author: your name
Date: 2021-03-17 18:27:25
LastEditTime: 2021-04-21 21:57:27
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /AttentionBasedNameDisambiguation/HeterogeneousGraph/localHANMetricLearning.py
'''

from HeterogeneousGraph.HAN import HAN
from utils import settings
from utils import data_utils, eval_utils
import codecs
from os.path import abspath, dirname, join
import numpy as np
from utils.cache import LMDBClient

def load_test_names():
    return data_utils.load_json(settings.DATA_DIR, 'test_name_list2.json')

def load_train_names():
    name_to_pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')
    return name_to_pubs_train

def testHAN():
    LMDB_NAME_EMB = settings.LMDB_NAME_EMB
    LMDB_PAPER_CENTER_EMB = settings.LMDB_PAPER_CENTER_EMB

    lc_emb = LMDBClient(LMDB_NAME_EMB)
    lc_emb_centers = LMDBClient(LMDB_PAPER_CENTER_EMB)
    han = HAN(lc_emb)

    name_to_pubs_train = load_train_names()
    for name in name_to_pubs_train:
        prec, rec, f1, pids, attentionEmbeddings, centers_embed_check = han.prepare_and_train(name=name, ispretrain=True, needtSNE=False)
        for pid, attentionEmbedding in zip(pids, attentionEmbeddings):
            lc_emb.set(pid, attentionEmbedding)
        
        for labelid, centerembedding in enumerate(centers_embed_check):
            lc_emb_centers.set(name+str(labelid), centerembedding)


        print (name, prec, rec, f1)
        break

def testUser(name):
    LMDB_NAME_EMB = settings.LMDB_NAME_EMB
    lc_emb = LMDBClient(LMDB_NAME_EMB)
    han = HAN(lc_emb)
    prec, rec, f1, pids, attentionEmbeddings, centers_embed_check = han.prepare_and_train(name=name, ispretrain=True, needtSNE=True)
    print (name, prec, rec, f1)

if __name__ == '__main__':
    testHAN()
    # name = "xiaofei_zhang"
    # name = "hai_yan_chen"
    # name = "gang_yin"
    # testUser(name)




