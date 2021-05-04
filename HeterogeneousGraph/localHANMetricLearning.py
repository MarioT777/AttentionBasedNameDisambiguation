'''
Author: your name
Date: 2021-03-17 18:27:25
LastEditTime: 2021-05-03 10:19:33
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
from HeterogeneousGraph import IDF_THRESHOLD


def load_test_names():
    name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json')
    return name_to_pubs_test

def load_train_names():
    name_to_pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')
    return name_to_pubs_train


def runTrainData():
    LMDB_NAME_EMB = settings.LMDB_NAME_EMB
    LMDB_PAPER_CENTER_EMB = settings.LMDB_PAPER_CENTER_EMB

    lc_emb = LMDBClient(LMDB_NAME_EMB)
    lc_emb_centers = LMDBClient(LMDB_PAPER_CENTER_EMB)
    han = HAN(lc_emb)

    name_to_pubs_train = load_train_names()
    for name in name_to_pubs_train:

        prec, rec, f1, pids, attentionEmbeddings, paper_center_embedd_check = han.prepare_and_train(name=name, ispretrain=True, needtSNE=False)
        for pid, attentionEmbedding in zip(pids, attentionEmbeddings):
            lc_emb.set(pid, attentionEmbedding)

        # centers_embed_check is a dict
        for (pid, paper_center_embedding) in paper_center_embedd_check.items():
            lc_emb_centers.set(pid, paper_center_embedding)

        print (name, prec, rec, f1)
    
def runTestData():
    LMDB_NAME_EMB = settings.LMDB_NAME_EMB
    LMDB_PAPER_CENTER_EMB = settings.LMDB_PAPER_CENTER_EMB

    lc_emb = LMDBClient(LMDB_NAME_EMB)
    lc_emb_centers = LMDBClient(LMDB_PAPER_CENTER_EMB)
    han = HAN(lc_emb)

    name_to_pubs_test = load_test_names()
    for name in name_to_pubs_test:

        prec, rec, f1, pids, attentionEmbeddings, paper_center_embedd_check = han.prepare_and_train(name=name, ispretrain=True, needtSNE=False)
        for pid, attentionEmbedding in zip(pids, attentionEmbeddings):
            lc_emb.set(pid, attentionEmbedding)

        # centers_embed_check is a dict
        for (pid, paper_center_embedding) in paper_center_embedd_check.items():
            lc_emb_centers.set(pid, paper_center_embedding)

        print (name, prec, rec, f1)

if __name__ == '__main__':
    runTrainData()
    runTestData()




