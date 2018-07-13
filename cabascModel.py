#!/usr/bin/env python
# encoding: utf-8

import os, sys
sys.path.append(os.getcwd())

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from nn_layer import dynamic_rnn, softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from att_layer import dot_produce_attention_layer, bilinear_attention_layer, mlp_attention_layer, Mlp_attention_layer, cam_mlp_attention_layer, mlp_layer, triple_attention_layer
from config import *
from utils import load_w2v, batch_index, load_inputs_cabasc



def cabasc(input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, ave_sent, mult_mask, sent_full, keep_prob1, keep_prob2, _id='all'):
    print('I am cabasc.')
    
    #CAM Module
    cell = tf.contrib.rnn.GRUCell
    # left GRU
    input_fw = tf.nn.dropout(input_fw, keep_prob1)
    hiddens_l = dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'leftGRU', 'all') #batch_size x N x d

    # right GRU
    input_bw = tf.nn.dropout(input_bw, keep_prob1)
    hiddens_r = dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'rightGRU', 'all') #batch_size x N x d

    #MLP layer for attention weight
    #left MLP
    beta_left = cam_mlp_attention_layer(hiddens_l, sen_len_fw, FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'bl') # batch_size x 1 x N
    beta_left = tf.squeeze(tf.nn.dropout(beta_left, keep_prob1)) # batch_size x N
    beta_left = tf.reverse(beta_left, [1])

    #right MLP
    beta_right = cam_mlp_attention_layer(hiddens_r, sen_len_bw, FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'br') # batch_size x 1 x N
    beta_right =  tf.squeeze(tf.nn.dropout(beta_right, keep_prob1)) # batch_size x N

    
    beta_add = tf.add(beta_left, beta_right) # batch_size x N
    beta = tf.multiply(beta_add, mult_mask) # batch_size x N
    beta = tf.expand_dims(beta, 2) # batch_size x N x 1
    beta_tiled = tf.tile(beta, [1, 1, FLAGS.embedding_dim])
    beta_3d = tf.reshape(beta_tiled, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim]) # batch_size x N x d

    # Memory Embedding
    M = tf.multiply(sent_full, beta_3d) # batch_size x N x d
    #M = sent_full # Model B

    # Target Embedding
    target_tiled = tf.tile(target,  [1, FLAGS.max_sentence_len])
    target_3d = tf.reshape(target_tiled, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim]) # batch_size x N x d

    # Average Target
    ave_sent_tiled = tf.tile(tf.expand_dims(ave_sent,1), [1, FLAGS.max_sentence_len, 1]) 
    ave_sent_3d = tf.reshape(ave_sent_tiled, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim]) # batch_size x N x d

    att_weights = triple_attention_layer(M, target_3d, ave_sent_3d, sen_len_fw + sen_len_bw - sen_len_tr, FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'eq7') # batch_size x 1 x N

    v_ts = tf.matmul(att_weights, M) # batch_size x 1 x d
    v_ns = tf.add(v_ts, tf.expand_dims(ave_sent,1)) # batch_size x 1 x d
    v_ns = tf.nn.dropout(v_ns, keep_prob1)

    v_ms = mlp_layer(v_ns, FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 'eq5') # batch_size x 1 x d

    prob = softmax_layer(tf.squeeze(v_ms), FLAGS.embedding_dim, FLAGS.random_base, keep_prob1, FLAGS.l2_reg, FLAGS.n_class)

    return prob, att_weights, beta_left, beta_right, v_ms


def main(train_path, test_path, accuracyOnt, test_size, remaining_size, learning_rate=FLAGS.learning_rate, keep_prob = FLAGS.keep_prob1):
    print_config()
    with tf.device('/gpu:1'):
        word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
        word_embedding = tf.constant(w2v, name='word_embedding')
        # word_embedding = tf.Variable(w2v, name='word_embedding')

        keep_prob1 = tf.placeholder(tf.float32, name="keep_prob1")
        keep_prob2 = tf.placeholder(tf.float32, name="keep_prob2")

        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name="x")
            y = tf.placeholder(tf.float32, [None, FLAGS.n_class], name="y")
            sen_len = tf.placeholder(tf.int32, None, name="sen_len")

            x_bw = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name="x_bw")
            sen_len_bw = tf.placeholder(tf.int32, [None], name="sen_len_bw")

            target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len], name="target_words")
            tar_len = tf.placeholder(tf.int32, [None], name="tar_len")

            sent_short = tf.placeholder(tf.int32, [None, None], name="sent_short")
            sent = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name="sent")
            mult_mask = tf.placeholder(tf.float32, [None, FLAGS.max_sentence_len], name="mult_mask")

        inputs_fw = tf.nn.embedding_lookup(word_embedding, x)       # batch x N x d
        inputs_bw = tf.nn.embedding_lookup(word_embedding, x_bw)    # batch x N x d
        sent_full = tf.nn.embedding_lookup(word_embedding, sent)    # batch x N x d
        
        #compute average sentence representation and target
        sentence_mask = tf.cast(tf.sequence_mask(sen_len_bw + sen_len - tar_len, FLAGS.max_sentence_len), tf.int32)
        sentence = tf.nn.embedding_lookup(word_embedding, tf.multiply(sent_short,sentence_mask))
        ave_sent = tf.divide(tf.reduce_sum(sentence, 1),tf.reshape(tf.tile(tf.cast(sen_len_bw + sen_len - tar_len, tf.float32),[FLAGS.embedding_dim]),[-1,FLAGS.embedding_dim]))
        
        target_mask = tf.cast(tf.sequence_mask(tar_len, FLAGS.max_target_len), tf.int32)
        target = tf.nn.embedding_lookup(word_embedding, tf.multiply(target_words,target_mask))
        target = tf.divide(tf.reduce_sum(target, 1),tf.reshape(tf.tile(tf.cast(tar_len, tf.float32), [FLAGS.embedding_dim]),[-1,FLAGS.embedding_dim]))


        # target = reduce_mean_with_len(target, tar_len)
        # for MLP & DOT
        # target = tf.expand_dims(target, 1)
        # batch_size = tf.shape(inputs_bw)[0]
        # target = tf.zeros([batch_size, FLAGS.max_sentence_len, FLAGS.embedding_dim]) + target
        # for BL
        # target = tf.squeeze(target)
        alpha_fw, alpha_bw = None, None
        prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r = cabasc(inputs_fw, inputs_bw, sen_len, sen_len_bw, target, tar_len, ave_sent, mult_mask, sent_full, keep_prob1, keep_prob2, 'all')

        loss = loss_func(y, prob)
        acc_num, acc_prob = acc_func(y, prob)
        global_step = tf.Variable(0, name='tr_global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
        # optimizer = train_func(loss, FLAGS.learning_rate, global_step)
        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(prob, 1)

        title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
            keep_prob,
            FLAGS.keep_prob2,
            FLAGS.batch_size,
            learning_rate,
            FLAGS.l2_reg,
            FLAGS.max_sentence_len,
            FLAGS.embedding_dim,
            FLAGS.n_hidden,
            FLAGS.n_class
        )

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        import time
        timestamp = str(int(time.time()))
        _dir = 'summary/' + str(timestamp) + '_' + title
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
        validate_summary_writer = summary_func(loss, acc_prob, test_loss, test_acc, _dir, title, sess)

        save_dir = 'temp_model/' + str(timestamp) + '_' + title + '/'
        # saver = saver_func(save_dir)

        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, '/-')

        if FLAGS.is_r == '1':
            is_r = True
        else:
            is_r = False

        tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, tr_sen_short, tr_sent, tr_mult_mask = load_inputs_cabasc(
            train_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )
        te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, te_sen_short, te_sent, te_mult_mask = load_inputs_cabasc(
            test_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )

        def get_batch_data(x_f, sen_len_f, x_b, sen_len_b, yi, target, tl, batch_size, kp1, kp2, senshort, sen, multMask, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                    
                feed_dict = {
                    x: x_f[index],
                    x_bw: x_b[index],
                    y: yi[index],
                    sen_len: sen_len_f[index],
                    sen_len_bw: sen_len_b[index],
                    target_words: target[index],
                    tar_len: tl[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                    sent_short: sen[index],
                    sent: sen[index],
                    mult_mask: multMask[index]
                }
                yield feed_dict, len(index)

        max_acc = 0.
        max_fw, max_bw = None, None
        max_tl, max_tr = None, None
        max_ty, max_py = None, None
        max_prob = None
        step = None
        for i in range(FLAGS.n_iter):
            trainacc, traincnt = 0., 0
            for train, numtrain in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len,
                                           FLAGS.batch_size, keep_prob, keep_prob, tr_sen_short, tr_sent, tr_mult_mask):
                # _, step = sess.run([optimizer, global_step], feed_dict=train)
                _, step, summary, _trainacc = sess.run([optimizer, global_step, train_summary_op, acc_num], feed_dict=train)
                train_summary_writer.add_summary(summary, step)
                # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                # sess.run(embed_update)
                trainacc += _trainacc            # saver.save(sess, save_dir, global_step=step)
                traincnt += numtrain
            print('finished train')
            acc, cost, cnt = 0., 0., 0
            fw, bw, tl, tr, ty, py = [], [], [], [], [], []
            p = []
            for test, num in get_batch_data(te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y,
                                            te_target_word, te_tar_len, 2000, 1.0, 1.0, te_sen_short, te_sent, te_mult_mask, False):
                if FLAGS.method == 'TD-ATT' or FLAGS.method == 'IAN':
                    _loss, _acc, _fw, _bw, _tl, _tr, _ty, _py, _p = sess.run(
                        [loss, acc_num, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r, true_y, pred_y, prob], feed_dict=test)
                    fw += list(_fw)
                    bw += list(_bw)
                    tl += list(_tl)
                    tr += list(_tr)
                else:
                    _loss, _acc, _ty, _py, _p, _fw = sess.run([loss, acc_num, true_y, pred_y, prob, alpha_fw], feed_dict=test)
                ty = np.asarray(_ty)
                py = np.asarray(_py)
                fw = np.asarray(_fw)
                p += list(_p)
                acc += _acc
                cost += _loss * num
                cnt += num
            print('all samples={}, correct prediction={}'.format(cnt, acc))
            trainacc = trainacc / traincnt
            acc = acc / cnt
            totalacc = ((acc * remaining_size) + (accuracyOnt * (test_size - remaining_size))) / test_size
            cost = cost / cnt
            print('Iter {}: mini-batch loss={:.6f}, train acc={:.6f}, test acc={:.6f}, combined acc={:.6f}'.format(i, cost,trainacc, acc, totalacc))
            summary = sess.run(test_summary_op, feed_dict={test_loss: cost, test_acc: acc})
            test_summary_writer.add_summary(summary, step)
            if acc > max_acc:
                max_acc = acc
                max_fw = fw
                max_bw = bw
                max_tl = tl
                max_tr = tr
                max_ty = ty
                max_py = py
                max_prob = p
        P = precision_score(max_ty, max_py, average=None)
        R = recall_score(max_ty, max_py, average=None)
        F1 = f1_score(max_ty, max_py, average=None)
        print('P:', P, 'avg=', sum(P) / FLAGS.n_class)
        print('R:', R, 'avg=', sum(R) / FLAGS.n_class)
        print('F1:', F1, 'avg=', sum(F1) / FLAGS.n_class)

        fp = open(FLAGS.prob_file, 'w')
        for item in max_prob:
            fp.write(' '.join([str(it) for it in item]) + '\n')
        fp = open(FLAGS.prob_file + '_fw', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_fw):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        fp = open(FLAGS.prob_file + '_bw', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_bw):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        fp = open(FLAGS.prob_file + '_tl', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_tl):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        fp = open(FLAGS.prob_file + '_tr', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_tr):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')

        print('Optimization Finished! Max acc={}'.format(max_acc))

        print('Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
            FLAGS.learning_rate,
            FLAGS.n_iter,
            FLAGS.batch_size,
            FLAGS.n_hidden,
            FLAGS.l2_reg
        ))
        return max_acc, np.where(np.subtract(max_py, max_ty) == 0, 0, 1), max_fw


if __name__ == '__main__':
    tf.app.run()
