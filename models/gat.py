import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN


class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                              out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]

        return logits

class HeteGAT_multi(BaseGAttN):

    def getCenters(num_classes, feature_size, labels, final_embed):

        #INF
        INF = 99999.0

        # this is wrong
        # centers = tf.zeros(shape=[num_classes, feature_size], dtype=tf.float32)
        #test
        centers = tf.Variable(tf.zeros([num_classes, feature_size]), dtype=tf.float32, name='centers')
        centers_count = tf.Variable(tf.zeros([num_classes, 1]), dtype=tf.float32, name='centers_count')
        # centers_count = centers_count + 1

        labels = tf.reshape(labels, [-1])

        # appear_times
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        appear_times = tf.cast(appear_times, tf.float32)

        centers = tf.scatter_add(centers, labels, final_embed)
        centers_count = tf.scatter_add(centers_count, labels, appear_times)
        centers_count = tf.clip_by_value(centers_count, clip_value_min=tf.constant(1.0, dtype=tf.float32), clip_value_max=tf.constant(INF,  dtype=tf.float32))
        # centers_count = tf.reciprocal(centers_count)

        # centers = tf.get_variable( shape=[num_classes, feature_size], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
        print ("====== getCenters =====")
        print ("labels:", labels)
        print ("final_embed: ", final_embed)
        # ====== （17，100） =====
        print ("centers: ", centers)
        print ("appear_times: ", appear_times)
        print ("centers_count: ", centers_count)
        print ("====== getCenters =====")

        # centers = centers * centers_count
        centers = centers / centers_count

        return centers

    def inference(inputs_list, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat_list, hid_units, n_heads, features, labels, activation=tf.nn.elu, residual=False,
                  mp_att_size=200, feature_size=100):

        #Metric Learning
        temp = inputs_list[0]
        # temp2 = tf.reduce_sum(temp, 0)
        # print ("temp2 check: ", temp2)
        MetricInputs = tf.layers.dense(temp, feature_size, activation=None)
        # ExpendMetricInputs = tf.expand_dims(MetricInputs, 0)
        print ("ExpendMetricInputs: check :", MetricInputs)
        inputs_list = [MetricInputs]


        # tests
        mp_att_size = 200
        embed_list = []
        for inputs, bias_mat in zip(inputs_list, bias_mat_list):
            attns = []
            jhy_embeds = []
            for _ in range(n_heads[0]):
                attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                              out_sz=hid_units[0], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
            h_1 = tf.concat(attns, axis=-1)

            for i in range(1, len(hid_units)):
                h_old = h_1
                attns = []
                for _ in range(n_heads[i]):
                    attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                                  out_sz=hid_units[i],
                                                  activation=activation,
                                                  in_drop=ffd_drop,
                                                  coef_drop=attn_drop, residual=residual))
                h_1 = tf.concat(attns, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))

        multi_embed = tf.concat(embed_list, axis=1)
        print ("multi_embed: ", multi_embed, ", mp_att_size: ", mp_att_size)
        final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        # feature_size, labels, features
        # num_classes, feature_size, labels, features


        # print("====== check labels and nbclass ======")
        # print(labels)
        # print(len(set(labels)))
        # print(nb_classes)
        # print("====== check labels and nbclass ======")


        centers_embed = HeteGAT_multi.getCenters(len(set(labels)), feature_size, labels, final_embed)
        
        # 尝试取消置换 
        centers_embed = tf.transpose(centers_embed)

        out = []
        for i in range(n_heads[-1]):
            out.append(tf.layers.dense(final_embed, nb_classes, activation=None))

        #     out.append(layers.attn_head(h_1, bias_mat=bias_mat,
        #                                 out_sz=nb_classes, activation=lambda x: x,
        #                                 in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
        # logits_list.append(logits)
        print('de')

        logits = tf.expand_dims(logits, axis=0)
        test_final_embeed = tf.reduce_sum(MetricInputs,0)
        return logits, final_embed , att_val, centers_embed, test_final_embeed
        # return logits, final_embed, att_val, centers_embed

class HeteGAT_no_coef(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat_list, hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=128):
        embed_list = []
        # coef_list = []
        for bias_mat in bias_mat_list:
            attns = []
            head_coef_list = []
            for _ in range(n_heads[0]):

                attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                                  out_sz=hid_units[0], activation=activation,
                                                  in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                                  return_coef=return_coef))
            h_1 = tf.concat(attns, axis=-1)
            for i in range(1, len(hid_units)):
                h_old = h_1
                attns = []
                for _ in range(n_heads[i]):
                    attns.append(layers.attn_head(h_1,
                                                  bias_mat=bias_mat,
                                                  out_sz=hid_units[i],
                                                  activation=activation,
                                                  in_drop=ffd_drop,
                                                  coef_drop=attn_drop,
                                                  residual=residual))
                h_1 = tf.concat(attns, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))


        # print('att for mp')
        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        # print(att_val)
        # last layer for clf
        out = []
        for i in range(n_heads[-1]):
            out.append(tf.layers.dense(final_embed, nb_classes, activation=None))
        #     out.append(layers.attn_head(h_1, bias_mat=bias_mat,
        #                                 out_sz=nb_classes, activation=lambda x: x,
        #                                 in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
        # logits_list.append(logits)
        print('de')
        logits = tf.expand_dims(logits, axis=0)
        # if return_coef:
        #     return logits, final_embed, att_val, coef_list
        # else:
        return logits, final_embed, att_val

class HeteGAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat_list, hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=128,
                  return_coef=False):
        embed_list = []
        coef_list = []
        for bias_mat in bias_mat_list:
            attns = []
            head_coef_list = []
            for _ in range(n_heads[0]):
                if return_coef:
                    a1, a2 = layers.attn_head(inputs, bias_mat=bias_mat,
                                              out_sz=hid_units[0], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                              return_coef=return_coef)
                    attns.append(a1)
                    head_coef_list.append(a2)
                    # attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                    #                               out_sz=hid_units[0], activation=activation,
                    #                               in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                    #                               return_coef=return_coef)[0])
                    #
                    # head_coef_list.append(layers.attn_head(inputs, bias_mat=bias_mat,
                    #                                        out_sz=hid_units[0], activation=activation,
                    #                                        in_drop=ffd_drop, coef_drop=attn_drop,
                    #                                        residual=False,
                    #                                        return_coef=return_coef)[1])
                else:
                    attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                                  out_sz=hid_units[0], activation=activation,
                                                  in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                                  return_coef=return_coef))
            head_coef = tf.concat(head_coef_list, axis=0)
            head_coef = tf.reduce_mean(head_coef, axis=0)
            coef_list.append(head_coef)
            h_1 = tf.concat(attns, axis=-1)
            for i in range(1, len(hid_units)):
                h_old = h_1
                attns = []
                for _ in range(n_heads[i]):
                    attns.append(layers.attn_head(h_1,
                                                  bias_mat=bias_mat,
                                                  out_sz=hid_units[i],
                                                  activation=activation,
                                                  in_drop=ffd_drop,
                                                  coef_drop=attn_drop,
                                                  residual=residual))
                h_1 = tf.concat(attns, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))
        # att for metapath
        # prepare shape for SimpleAttLayer
        # print('att for mp')
        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        # print(att_val)
        # last layer for clf
        out = []
        for i in range(n_heads[-1]):
            out.append(tf.layers.dense(final_embed, nb_classes, activation=None))
        #     out.append(layers.attn_head(h_1, bias_mat=bias_mat,
        #                                 out_sz=nb_classes, activation=lambda x: x,
        #                                 in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
        # logits_list.append(logits)
        logits = tf.expand_dims(logits, axis=0)
        if return_coef:
            return logits, final_embed, att_val, coef_list
        else:
            return logits, final_embed, att_val
        
 
