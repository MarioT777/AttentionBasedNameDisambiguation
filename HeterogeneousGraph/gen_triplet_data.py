from os.path import join
import os
import multiprocessing as mp
import random
from datetime import datetime
from utils.cache import LMDBClient
from utils import data_utils
from utils import settings

# LMDB_NAME = "author_100.emb.weighted"
LMDB_NAME = "lc_attention_network_embedding2"
lc = LMDBClient(LMDB_NAME)

rawFeatureEmbedding = "author_100.emb.weighted"
lc2 = LMDBClient(rawFeatureEmbedding)


start_time = datetime.now()

"""
This class generates triplets of author embeddings to train global_ model
"""


class TripletsGenerator:
    name2pubs_train = {}
    name2pubs_test = {}
    names_train = None
    names_test = None
    n_pubs_train = None
    n_pubs_test = None
    pids_train = []
    pids_test = []
    n_triplets = 0
    batch_size = 100000

    def __init__(self, train_scale=10000):
        self.prepare_data()
        self.save_size = train_scale
        self.idf = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'feature_idf.pkl')

    def prepare_data(self):
        self.name2pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')  # for test
        self.name2pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json')
        self.names_train = self.name2pubs_train.keys()
        # print('names train', len(self.names_train))
        self.names_test = self.name2pubs_test.keys()
        # print('names test', len(self.names_test))
        assert not set(self.names_train).intersection(set(self.names_test))
        for name in self.names_train:
            name_pubs_dict = self.name2pubs_train[name]
            for aid in name_pubs_dict:
                self.pids_train += name_pubs_dict[aid]
        random.shuffle(self.pids_train)
        self.n_pubs_train = len(self.pids_train)
        # print('pubs2train', self.n_pubs_train)

        for name in self.names_test:
            name_pubs_dict = self.name2pubs_test[name]
            for aid in name_pubs_dict:
                self.pids_test += name_pubs_dict[aid]
        random.shuffle(self.pids_test)
        self.n_pubs_test = len(self.pids_test)
        # print('pubs2test', self.n_pubs_test)

    def gen_neg_pid(self, not_in_pids, role='train'):
        if role == 'train':
            sample_from_pids = self.pids_train
        else:
            sample_from_pids = self.pids_test
        while True:
            idx = random.randint(0, len(sample_from_pids)-1)
            pid = sample_from_pids[idx]
            if pid not in not_in_pids:
                return pid

    def sample_triplet_ids(self, task_q, role='train', N_PROC=8):
        n_sample_triplets = 0
        if role == 'train':
            names = self.names_train
            name2pubs = self.name2pubs_train
        else:  # test
            names = self.names_test
            name2pubs = self.name2pubs_test
            self.save_size = 200000  # test save size
        for name in names:
            name_pubs_dict = name2pubs[name]
            for aid in name_pubs_dict:
                pub_items = name_pubs_dict[aid]
                #
                # try it
                if len(pub_items) == 1:
                    continue

                pids = pub_items
                cur_n_pubs = len(pids)
                random.shuffle(pids)
                for i in range(cur_n_pubs):
                    pid1 = pids[i]  # pid

                    # batch samples
                    n_samples_anchor = min(6, cur_n_pubs)
                    idx_pos = random.sample(range(cur_n_pubs), n_samples_anchor)
                    for ii, i_pos in enumerate(idx_pos):
                        if i_pos != i:
                            if n_sample_triplets % 100 == 0:
                                # print('sampled triplet ids', n_sample_triplets)
                                pass
                            pid_pos = pids[i_pos]
                            pid_neg = self.gen_neg_pid(pids, role)
                            n_sample_triplets += 1
                            task_q.put((pid1, pid_pos, pid_neg))

                            if n_sample_triplets >= self.save_size:
                                for j in range(N_PROC):
                                    task_q.put((None, None, None))
                                return
        for j in range(N_PROC):
            task_q.put((None, None, None))

    def getLMDBEmbedding(self, pid):
        return lc.get(pid)

    def getAnchorEmbedding(self, pid):
        return lc2.get(pid)

    def embeddings(self, anchorPid, pid_pos, pid_neg):
        if self.getLMDBEmbedding(anchorPid) is not None and  self.getLMDBEmbedding(pid_pos) is not None and self.getLMDBEmbedding(pid_neg) is not None \
                and self.getAnchorEmbedding(anchorPid) is not None and self.getAnchorEmbedding(pid_pos) is not None and self.getAnchorEmbedding(pid_neg) is not None:
            return self.getAnchorEmbedding(anchorPid), self.getAnchorEmbedding(pid_pos), self.getAnchorEmbedding(pid_neg), self.getLMDBEmbedding(anchorPid), self.getLMDBEmbedding(pid_pos), self.getLMDBEmbedding(pid_neg)
        else:
            return None, None, None, None, None, None

    def testembeddings(self, anchorPid, pid_pos, pid_neg):
        return self.getAnchorEmbedding(anchorPid), self.getAnchorEmbedding(pid_pos), self.getAnchorEmbedding(pid_neg), self.getAnchorEmbedding(anchorPid), self.getAnchorEmbedding(pid_pos), self.getAnchorEmbedding(pid_neg)

    def gen_emb_mp(self, task_q, emb_q, role):
        while True:
            pid1, pid_pos, pid_neg = task_q.get()
            if pid1 is None:
                break
            # emb1 = self.getAnchorEmbedding(pid1)
            # emb_pos = self.getAnchorEmbedding(pid_pos)
            # emb_neg = self.getAnchorEmbedding(pid_neg)
            if role == "test":
                emb1, emb_pos, emb_neg, attentionEmb, attentionEmbPos, attentionEmbNeg  = self.testembeddings(pid1, pid_pos, pid_neg)
            else:
                emb1, emb_pos, emb_neg, attentionEmb, attentionEmbPos, attentionEmbNeg = self.embeddings(pid1, pid_pos, pid_neg)
            # emb1 = lc.get(pid1)
            # emb_pos = lc.get(pid_pos)
            # emb_neg = lc.get(pid_neg)
            if emb1 is not None and emb_pos is not None and emb_neg is not None and attentionEmb is not None:
                emb_q.put((emb1, emb_pos, emb_neg, attentionEmb, attentionEmbPos, attentionEmbNeg))
        emb_q.put((False, False, False, False, False, False))

    def gen_triplets_mp(self, role='train'):
        N_PROC = 8

        task_q = mp.Queue(N_PROC * 6)
        emb_q = mp.Queue(1000)

        producer_p = mp.Process(target=self.sample_triplet_ids, args=(task_q, role, N_PROC))
        consumer_ps = [mp.Process(target=self.gen_emb_mp, args=(task_q, emb_q, role)) for _ in range(N_PROC)]
        producer_p.start()
        [p.start() for p in consumer_ps]

        cnt = 0

        while True:
            # if cnt % 1000 == 0:
                # print('get', cnt, datetime.now()-start_time)
            emb1, emb_pos, emb_neg, attentionEmb, attentionEmbPos, attentionEmbNeg = emb_q.get()
            if emb1 is False:
                producer_p.terminate()
                producer_p.join()
                [p.terminate() for p in consumer_ps]
                [p.join() for p in consumer_ps]
                break
            cnt += 1
            yield (emb1, emb_pos, emb_neg, attentionEmb, attentionEmbPos, attentionEmbNeg)

    def dump_triplets(self, role='train'):
        triplets = self.gen_triplets_mp(role)
        if role == 'train':
            out_dir = join(settings.OUT_DIR, 'triplets-{}'.format(self.save_size))
        else:
            out_dir = join(settings.OUT_DIR, 'test-triplets')
        os.makedirs(out_dir, exist_ok=True)
        anchor_embs = []
        pos_embs = []
        neg_embs = []
        atten_embs = []
        attenPos_embs = []
        attenNeg_embs = []
        f_idx = 0
        for i, t in enumerate(triplets):
            # if i % 100 == 0:
                # print(i, datetime.now()-start_time)
            emb_anc, emb_pos, emb_neg, emb_atten, attentionEmbPos, attentionEmbNeg = t[0], t[1], t[2], t[3], t[4], t[5]
            anchor_embs.append(emb_anc)
            pos_embs.append(emb_pos)
            neg_embs.append(emb_neg)
            atten_embs.append(emb_atten)
            attenPos_embs.append(attentionEmbPos)
            attenNeg_embs.append(attentionEmbNeg)
            if len(anchor_embs) == self.batch_size:
                data_utils.dump_data(anchor_embs, out_dir, 'anchor_embs_{}_{}.pkl'.format(role, f_idx))
                data_utils.dump_data(pos_embs, out_dir, 'pos_embs_{}_{}.pkl'.format(role, f_idx))
                data_utils.dump_data(neg_embs, out_dir, 'neg_embs_{}_{}.pkl'.format(role, f_idx))
                data_utils.dump_data(atten_embs, out_dir, 'atten_embs_{}_{}.pkl'.format(role, f_idx))
                data_utils.dump_data(attenPos_embs, out_dir, 'atten_pos_{}_{}.pkl'.format(role, f_idx))
                data_utils.dump_data(attenNeg_embs, out_dir, 'atten_neg_{}_{}.pkl'.format(role, f_idx))
                f_idx += 1
                anchor_embs = []
                pos_embs = []
                neg_embs = []
        if anchor_embs:
            data_utils.dump_data(anchor_embs, out_dir, 'anchor_embs_{}_{}.pkl'.format(role, f_idx))
            data_utils.dump_data(pos_embs, out_dir, 'pos_embs_{}_{}.pkl'.format(role, f_idx))
            data_utils.dump_data(neg_embs, out_dir, 'neg_embs_{}_{}.pkl'.format(role, f_idx))
            data_utils.dump_data(atten_embs, out_dir, 'atten_embs_{}_{}.pkl'.format(role, f_idx))
            data_utils.dump_data(attenPos_embs, out_dir, 'atten_pos_{}_{}.pkl'.format(role, f_idx))
            data_utils.dump_data(attenNeg_embs, out_dir, 'atten_neg_{}_{}.pkl'.format(role, f_idx))

        print('dumped')


if __name__ == '__main__':
    data_gen = TripletsGenerator(train_scale=1000000)
    data_gen.dump_triplets(role='train')
    data_gen.dump_triplets(role='test')
