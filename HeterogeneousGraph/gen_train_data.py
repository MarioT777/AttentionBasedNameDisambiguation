from os.path import join
import os
from utils import data_utils, inputData
from utils import settings
from utils.cache import LMDBClient
from HeterogeneousGraph import IDF_THRESHOLD, Author_THRESHOLD
from collections import defaultdict
from global_.global_model import GlobalTripletModel
import numpy as np
from utils.eval_utils import get_hidden_output

def dump_inter_emb():
    """
    dump hidden embedding via trained global_ model for local model to use
    """
    Res = defaultdict(list)
    LMDB_NAME = "author_100.emb.weighted"
    lc_input = LMDBClient(LMDB_NAME)
    INTER_LMDB_NAME = 'author_triplets.emb'
    lc_inter = LMDBClient(INTER_LMDB_NAME)
    global_model = GlobalTripletModel(data_scale=1000000)
    trained_global_model = global_model.load_triplets_model()
    name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json')
    # print(name_to_pubs_test)
    for name in name_to_pubs_test:
        name_data = name_to_pubs_test[name]
        embs_input = []
        pids = []
        for i, aid in enumerate(name_data.keys()):
            if len(name_data[aid]) < 5:  # n_pubs of current author is too small
                continue
            for pid in name_data[aid]:
                cur_emb = lc_input.get(pid)
                if cur_emb is None:
                    continue
                embs_input.append(cur_emb)
                pids.append(pid)
        embs_input = np.stack(embs_input)
        inter_embs = get_hidden_output(trained_global_model, embs_input)
        for i, pid_ in enumerate(pids):
            lc_inter.set(pid_, inter_embs[i])
            Res[pid_].append(inter_embs[i])

    print(Res)

class DataGenerator:
    pids_train = []
    AuthorSocial = None
    def __init__(self):
        self.AuthorSocial = inputData.loadAuthorSocial()


    def prepare_data(self):
        self.name2pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')  # for test
        self.name2pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json')
        self.names_train = self.name2pubs_train.keys()
        self.names_test = self.name2pubs_test.keys()
        assert not set(self.names_train).intersection(set(self.names_test))

        for authorName in self.names_train:
            self.genPAPandPSP(authorName=authorName, idf_threshold=IDF_THRESHOLD)


    def genPAPandPSP(self, authorName="hongbin_li", idf_threshold=10):
        idf = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'feature_idf.pkl')
        raw_word2vec = 'author_100.emb.weighted'
        lc_emb = LMDBClient(raw_word2vec)
        LMDB_AUTHOR_FEATURE = "pub_authors.feature"
        lc_feature = LMDBClient(LMDB_AUTHOR_FEATURE)
        cur_person_dict = self.name2pubs_train[authorName]
        pids_set = set()
        pids = []
        pids2label = {}

        print ("pass0")
        graph_dir = join(settings.DATA_DIR, 'AttentionNetwork' , 'graph-{}'.format(idf_threshold))
        # generate content
        wf_content = open(join(graph_dir, '{}_feature_and_label.txt'.format(authorName)), 'w')
        for i, aid in enumerate(cur_person_dict):
            personPids = cur_person_dict[aid]
            # It's no nessary to use these data which's length is less than 5
            if len(personPids) < 5:
                continue
            print ("aid: ", aid, ", pids: ", pids)
            for pid in personPids:
                pids2label[str(pid)] = str(aid)
                pids.append(pid)

        print ("pass1")
        for pid in pids:
            # use raw feature rather than Triplet Loss
            cur_pub_emb = lc_emb.get(pid)
            # cur_pub_emb = lc_inter.get(pid)
            if cur_pub_emb is not None:
                cur_pub_emb = list(map(str, cur_pub_emb))
                pids_set.add(pid)
                wf_content.write('{}\t'.format(pid))
                wf_content.write('\t'.join(cur_pub_emb))
                wf_content.write('\t{}\n'.format(pids2label[pid]))
        wf_content.close()

        print ("pass2")
        # generate network1
        pids_filter = list(pids_set)
        n_pubs = len(pids_filter)
        print('n_pubs', n_pubs)
        wf_network = open(join(graph_dir, '{}_PAP.txt'.format(authorName)), 'w')
        for i in range(n_pubs-1):
            if i % 10 == 0:
                print(i)
            author_feature1 = set(lc_feature.get(pids_filter[i]))
            for j in range(i+1, n_pubs):
                author_feature2 = set(lc_feature.get(pids_filter[j]))
                # print('author_feature2: ', author_feature2)
                common_features = author_feature1.intersection(author_feature2)
                idf_sum = 0
                for f in common_features:
                    idf_sum += idf.get(f, idf_threshold)
                    # print(f, idf.get(f, idf_threshold))
                if idf_sum >= idf_threshold:
                    wf_network.write('{}\t{}\n'.format(pids_filter[i], pids_filter[j]))
        wf_network.close()

        def CountNumber(A, B):
            res = 0
            for x in A:
                for y in B:
                    if x == y:
                        res = res + 1

            return res

        print ("pass3")
        wf_network = open(join(graph_dir, '{}_PSP.txt'.format(authorName)), 'w')

        for i in range(n_pubs-1):
            for j in range(i + 1, n_pubs):
                Graph1Socials = self.AuthorSocial[pids_filter[i]]
                Graph2Socials = self.AuthorSocial[pids_filter[j]]
                if CountNumber(Graph1Socials, Graph2Socials) >= Author_THRESHOLD:
                    wf_network.write('{}\t{}\n'.format(pids_filter[i], pids_filter[j]))

        wf_network.close()

    def constructNetwork(self, authorName):
        name_pubs_dict = self.name2pubs_train[authorName]
        for aid in name_pubs_dict:
            self.pids_train += name_pubs_dict[aid]


    def run(self):
        self.prepare_data()


if __name__ == '__main__':
    datagenerator = DataGenerator()
    datagenerator.prepare_data()