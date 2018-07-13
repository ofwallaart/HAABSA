#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf


def softmax_with_len(inputs, length, max_len):
    inputs = tf.cast(inputs, tf.float32)
    # max_axis = tf.reduce_max(inputs, -1, keep_dims=True)
    # inputs = tf.exp(inputs - max_axis)
    inputs = tf.exp(inputs)
    length = tf.reshape(length, [-1])
    mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
    inputs *= mask
    _sum = tf.reduce_sum(inputs, reduction_indices=-1, keep_dims=True) + 1e-9
    return inputs / _sum


def bilinear_attention_layer(inputs, attend, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param attend: batch * n_hidden
    :param length:
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id:
    :return:
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.reshape(tf.matmul(inputs, w), [-1, max_len, n_hidden])
    attend = tf.expand_dims(attend, 2)
    tmp = tf.reshape(tf.matmul(tmp, attend), [batch_size, 1, max_len])
    # M = tf.expand_dims(tf.matmul(attend, w), 2)
    # tmp = tf.reshape(tf.batch_matmul(inputs, M), [batch_size, 1, max_len])
    return softmax_with_len(tmp, length, max_len)


def dot_produce_attention_layer(inputs, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param length: batch * 1
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id: layer's identical id
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    u = tf.get_variable(
        name='att_u_' + str(layer_id),
        shape=[n_hidden, 1],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + 1))),
        # initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + 1)), np.sqrt(6.0 / (n_hidden + 1))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.reshape(tf.matmul(inputs, u), [batch_size, 1, max_len])
    alpha = softmax_with_len(tmp, length, max_len)
    return alpha


def mlp_attention_layer(inputs, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param length: batch * 1
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id: layer's identical id
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='att_b' + str(layer_id),
        shape=[n_hidden],
        # initializer=tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-0., 0.),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    u = tf.get_variable(
        name='att_u_' + str(layer_id),
        shape=[n_hidden, 1],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + 1))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + 1)), np.sqrt(6.0 / (n_hidden + 1))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.tanh(tf.matmul(inputs, w) + b)
    tmp = tf.reshape(tf.matmul(tmp, u), [batch_size, 1, max_len])
    alpha = softmax_with_len(tmp, length, max_len)
    return alpha

def cam_mlp_attention_layer(inputs, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param length: batch * 1
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id: layer's identical id
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, 1],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='att_b' + str(layer_id),
        shape=[1],
        # initializer=tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-0., 0.),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.sigmoid(tf.matmul(inputs, w) + b) + 0.5
    tmp = tf.reshape(tmp, [batch_size, 1, max_len])
    # mask = tf.reshape(tf.reverse(tf.cast(tf.sequence_mask(length, max_len), tf.float32),[1]), tf.shape(tmp))
    # alpha = tf.multiply(tmp, mask)
    return tmp


def Mlp_attention_layer(inputs, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param length: batch * 1
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id: layer's identical id
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        # initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    u = tf.get_variable(
        name='att_u_' + str(layer_id),
        shape=[n_hidden, 1],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + 1))),
        # initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + 1)), np.sqrt(6.0 / (n_hidden + 1))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.transpose(tf.reshape(inputs, [-1, n_hidden]), [1, 0])
    tmp = tf.transpose(tf.tanh(tf.matmul(w, inputs)), [1, 0])
    tmp = tf.reshape(tf.matmul(tmp, u), [batch_size, 1, max_len])
    alpha = softmax_with_len(tmp, length, max_len)
    return alpha

def triple_attention_layer(input1, input2, input3, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param input1, input2, input3: batch * max_len * n_hidden
    :param length: batch * 1
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id: layer's identical id
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(input1)[0]
    max_len = tf.shape(input1)[1]
    w6 = tf.get_variable(
        name='att_w6_' + str(layer_id),
        shape=[n_hidden, 1],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    w7 = tf.get_variable(
        name='att_w7_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )

    w8 = tf.get_variable(
        name='att_w8_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    w9 = tf.get_variable(
        name='att_w9_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='att_b' + str(layer_id),
        shape=[n_hidden],
        # initializer=tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-0., 0.),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    input1 = tf.reshape(input1, [-1, n_hidden])
    input2 = tf.reshape(input2, [-1, n_hidden])
    input3 = tf.reshape(input3, [-1, n_hidden])
    tmp = tf.tanh(tf.matmul(input1, w7) + tf.matmul(input2, w8) + tf.matmul(input3, w9) + b)
    tmp = tf.matmul(tmp, w6)
    tmp = tf.reshape(tmp, [batch_size, 1, max_len])
    alpha = softmax_with_len(tmp, length, max_len)
    return alpha

def mlp_layer(inputs, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * 1 * n_hidden
    :param length: batch * 1
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id: layer's identical id
    :return: batch * 1 * n_hidden
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='att_b' + str(layer_id),
        shape=[1, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-0., 0.),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.tanh(tf.matmul(inputs, w) + b)
    tmp = tf.reshape(tmp, [batch_size, 1, n_hidden])
    alpha = tmp
    return alpha

def triple_attention_layer2(input1, input2, input3, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param input1, input2, input3: batch * max_len * n_hidden
    :param length: batch * 1
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id: layer's identical id
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(input1)[0]
    max_len = tf.shape(input1)[1]
    w6 = tf.get_variable(
        name='att_w6_' + str(layer_id),
        shape=[n_hidden, 1],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    w7 = tf.get_variable(
        name='att_w7_' + str(layer_id),
        shape=[2*n_hidden, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )

    w8 = tf.get_variable(
        name='att_w8_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    w9 = tf.get_variable(
        name='att_w9_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='att_b' + str(layer_id),
        shape=[n_hidden],
        # initializer=tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-0., 0.),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    input1 = tf.reshape(input1, [-1, 2*n_hidden])
    input2 = tf.reshape(input2, [-1, n_hidden])
    input3 = tf.reshape(input3, [-1, n_hidden])
    tmp = tf.tanh(tf.matmul(input1, w7) + tf.matmul(input2, w8) + tf.matmul(input3, w9) + b)
    # tmp = tf.matmul(tmp, w6)
    tmp = tf.reshape(tmp, [batch_size, 1, n_hidden])
    alpha = tf.nn.softmax(tmp)
    return alpha

def mlp_layer2(inputs, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * n_hidden
    :param length: batch * 1
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id: layer's identical id
    :return: batch * n_hidden
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='att_b' + str(layer_id),
        shape=[1, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-0., 0.),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.tanh(tf.matmul(inputs, w) + b)
    tmp = tf.reshape(tmp, [batch_size, n_hidden])
    alpha = tmp
    return alpha