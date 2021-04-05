'''
Author: your name
Date: 2021-03-17 18:27:25
LastEditTime: 2021-03-18 11:52:12
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /AttentionBasedNameDisambiguation/utils/settings.py
'''
from os.path import abspath, dirname, join
import os

PROJ_DIR = join(abspath(dirname(__file__)), '..')

DATA_DIR = join(PROJ_DIR, 'data')
OUT_DIR = join(PROJ_DIR, 'out')
PIC_DIR = join(PROJ_DIR, 'pic')
EMB_DATA_DIR = join(DATA_DIR, 'emb')
GLOBAL_DATA_DIR = join(DATA_DIR, 'global')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EMB_DATA_DIR, exist_ok=True)
