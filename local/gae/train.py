from __future__ import division
from __future__ import print_function

import os
import time
from os.path import join
from utils import settings
# from global_.prepare_local_data import IDF_THRESHOLD

IDF_THRESHOLD = 32
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
local_na_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(IDF_THRESHOLD))


import codecs
import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from local.gae.optimizer import OptimizerAE, OptimizerVAE
from local.gae.model import GCNModelAE, GCNModelVAE
from local.gae.preprocessing import preprocess_graph, construct_feed_dict, \
    sparse_to_tuple, normalize_vectors, gen_train_edges, cal_pos_weight
from utils.cluster import clustering
from utils.data_utils import load_json
from utils.eval_utils import pairwise_precision_recall_f1, cal_f1
from utils import settings, tSNEAnanlyse

from sklearn.metrics.cluster import normalized_mutual_info_score

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 150, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')  # 32
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')  # 16
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('name', 'hui_fang', 'Dataset string.')
# flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('is_sparse', 0, 'Whether input features are sparse.')

model_str = FLAGS.model
name_str = FLAGS.name
start_time = time.time()

from utils.cache import LMDBClient
INTER_LMDB_NAME = 'triplete_loss_lc_attention_network_embedding'
lc_inter = LMDBClient(INTER_LMDB_NAME)

RAW_INTER_NAME = 'author_100.emb.weighted'
lc_inter_raw = LMDBClient(INTER_LMDB_NAME)

tripleteLossLMDBName = 'author_triplets.emb'
tripletFeature = LMDBClient(tripleteLossLMDBName)


RAWFEATURE = "rawfeature"
ATTENTIONFEATURE = "attention_feature"
TRIPLETFEATURE = "triplet_feature"

def encode_labels(labels):
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    return list(map(lambda x: classes_dict[x], labels))

def load_local_data(path=local_na_dir, name='cheng_cheng', rawfeature=False):
    # Load local paper network dataset
    print('Loading {} dataset...'.format(name), 'path=', path, name)

    idx_features_labels = np.genfromtxt(join(path , "{}_pubs_content.txt".format(name)), dtype=np.dtype(str))
    # features = np.array(idx_features_labels[:, 1:-2], dtype=np.float32)  # sparse?
    labels = encode_labels(idx_features_labels[:, -2])

    features = []
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.str)
    for id in idx:

        if rawfeature == ATTENTIONFEATURE:
            features.append(lc_inter.get(id))
        elif rawfeature == RAW_INTER_NAME:
            features.append(lc_inter_raw.get(id))
        else:
            features.append(tripletFeature.get(id))


    features = np.array(features, dtype=np.float32)

    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(join(path, "{}_pubs_network.txt".format(name)), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return adj, features, labels

def gae_for_na(name, rawfeature, local_data_path=join(settings.DATA_DIR, 'local', 'graph-{}'.format(IDF_THRESHOLD))):
    """
    train and evaluate disambiguation results for a specific name
    :param name:  author name
    :return: evaluation results
    """
    adj, features, labels = load_local_data(path=local_data_path, name=name, rawfeature=rawfeature)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_train = gen_train_edges(adj)

    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    num_nodes = adj.shape[0]
    input_feature_dim = features.shape[1]
    if FLAGS.is_sparse:  # TODO to test
        # features = sparse_to_tuple(features.tocoo())
        # features_nonzero = features[1].shape[0]
        features = features.todense()  # TODO
    else:
        features = normalize_vectors(features)

    # Define placeholders
    placeholders = {
        # 'features': tf.sparse_placeholder(tf.float32),
        'features': tf.placeholder(tf.float32, shape=(None, input_feature_dim)),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, input_feature_dim)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, input_feature_dim, num_nodes)
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()  # negative edges/pos edges
    print('positive edge weight', pos_weight)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.nnz) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    def get_embs():
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)  # z_mean is better
        return emb

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                        feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "train_acc=", "{:.5f}".format(avg_accuracy),
              "time=", "{:.5f}".format(time.time() - t))

    emb = get_embs()
    n_clusters = len(set(labels))
    emb_norm = normalize_vectors(emb)
    clusters_pred = clustering(emb_norm, num_clusters=n_clusters)
    prec, rec, f1 =  pairwise_precision_recall_f1(clusters_pred, labels)

    NMI1 = normalized_mutual_info_score(clusters_pred, labels)

    print('pairwise precision', '{:.5f}'.format(prec),
          'recall', '{:.5f}'.format(rec),
          'f1', '{:.5f}'.format(f1),
          'NMI : ', '{:.5f}'.format(NMI1))

    clusters_pred2 = clustering(features, num_clusters=n_clusters)
    prec2, rec2, f12 =  pairwise_precision_recall_f1(clusters_pred2, labels)

    NMI2 = normalized_mutual_info_score(clusters_pred2, labels)
    print('pairwise precision', '{:.5f}'.format(prec2),
          'recall', '{:.5f}'.format(rec2),
          'f1', '{:.5f}'.format(f12),
          'NMI : ', '{:.5f}'.format(NMI2))

    from sklearn.manifold import TSNE
    features_new = TSNE(learning_rate=100).fit_transform(features)
    emb_new = TSNE(learning_rate=100).fit_transform(emb_norm)

    labels = np.array(labels) + 2
    clusters_pred = np.array(clusters_pred) + 2
    clusters_pred2 = np.array(clusters_pred2) + 2


    if rawfeature == RAW_INTER_NAME:
        tSNEAnanlyse(emb_norm, labels, join(settings.PIC_DIR, "FINALResult", "rawReature_%s_gae_final_raw.png" % (name)))
        tSNEAnanlyse(features, labels, join(settings.PIC_DIR, "FINALResult", "rawReature_%s_gae_features_raw.png" % (name)))
    elif rawfeature == ATTENTIONFEATURE:
        tSNEAnanlyse(emb_new, labels, join(settings.PIC_DIR, "FINALResult", "rawReature_%s_gae_final.png" % (name)))
        tSNEAnanlyse(features_new, labels, join(settings.PIC_DIR, "FINALResult", "rawReature_%s_gae_features.png" % (name)))
        tSNEAnanlyse(emb_new, clusters_pred, join(settings.PIC_DIR, "FINALResult", "rawReature_%s_gae_final_clusterresult.png" % (name)))
        tSNEAnanlyse(features_new, clusters_pred2, join(settings.PIC_DIR, "FINALResult", "rawReature_%s_gae_features_clusterresult.png" % (name)))
    else:
        tSNEAnanlyse(emb_norm, labels, join(settings.PIC_DIR, "FINALResult", "rawReature_%s_gae_final_triplet.png" % (name)))
        tSNEAnanlyse(features, labels, join(settings.PIC_DIR, "FINALResult", "rawReature_%s_gae_features_triplet.png" % (name)))

    return [prec, rec, f1, prec2, rec2, f12, NMI1, NMI2], num_nodes, n_clusters


def load_test_names():
    return load_json(settings.DATA_DIR, 'test_name_list.json')


def main():
    names = load_test_names()
    wf = codecs.open(join(settings.OUT_DIR, 'local_clustering_results.csv'), 'w', encoding='utf-8')
    wf.write('name,n_pubs,n_clusters,precision,recall,f1, NMI\n')
    metrics = np.zeros(8)
    cnt = 0

    macro_prec_avg = 0
    macro_rec_avg = 0
    macro_f1_avg = 0
    nmi_avg = 0

    for name in names:
        cur_metric, num_nodes, n_clusters = gae_for_na(name, rawfeature="attention_feature")
        # cur_metric, num_nodes, n_clusters = gae_for_na(name, rawfeature="attention_feature")
            
        wf.write('{0},{1},{2},{3:.5f},{4:.5f},{5:.5f}, {6:.5f}\n'.format(
            name, num_nodes, n_clusters, cur_metric[0], cur_metric[1], cur_metric[2] , cur_metric[6]  ))
        
        wf.write('{0},{1},{2},{3:.5f},{4:.5f},{5:.5f}, {6:.5f}\n'.format(
            name + str('_raw_feature'), num_nodes, n_clusters, cur_metric[3], cur_metric[4], cur_metric[5], cur_metric[7]))

        wf.flush()
        for i, m in enumerate(cur_metric):
            metrics[i] += m
        cnt += 1
        macro_prec = metrics[0] / cnt
        macro_rec = metrics[1] / cnt
        macro_nmi = metrics[6] / cnt

        macro_f1 = cal_f1(macro_prec, macro_rec)

        macro_prec_avg += macro_prec
        macro_rec_avg += macro_rec_avg
        macro_f1_avg += macro_f1
        nmi_avg += macro_nmi

        print('average until now', [macro_prec, macro_rec, macro_f1, macro_nmi])
        time_acc = time.time()-start_time
        print(cnt, 'names', time_acc, 'avg time', time_acc/cnt)
    
    macro_prec_ = macro_prec_avg / cnt
    macro_rec = macro_rec_avg / cnt
    macro_f1 = macro_f1_avg / cnt

    wf.write('average,,,{0:.5f},{1:.5f},{2:.5f}, {3:.5f}\n'.format(
        macro_prec, macro_rec, macro_f1, macro_nmi))
    wf.close()




if __name__ == '__main__':
    # gae_for_na('hongbin_li')
    # gae_for_na('j_yu')
    # kexin_xu
    # author = 'hongbin_li'
    # author = 'kexin_xu'
    # Res1 = gae_for_na(author, rawfeature="rawfeature")
    # Res2 = gae_for_na(author, rawfeature="attention_feature")
    # Res3 = gae_for_na(author, rawfeature="triplet_feature")
    # print ("raw feature: ", Res1)
    # print ("not raw feature: ", Res2)
    # print ("triplet raw feature: ", Res3)
    main()


# 650 hongbin_li_pubs_network.txt
# 9459 hongbin_li_pubs_network.txt

# 12987 kexin_xu_pubs_network.txt
# 1354 data/local/graph-32/kexin_xu_pubs_network.txt
# 2671 kexin_xu_pubs_network.txt
# 5712 data/local/graph-32/kexin_xu_pubs_network.txt


# average until now [0.7509695493338227, 0.6284136254612461, 0.6842471413760391]