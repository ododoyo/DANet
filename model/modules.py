import os, sys
libpath = os.path.abspath("..")
sys.path.append(libpath)

import itertools
import numpy as np
import tensorflow as tf

class Encoder(object):
    def __init__(self, cfg, name=None):
        self._cfg = cfg
        self.name = name

    def encode(self, s_mixed, seq_lens, fw_drop_keep, recur_drop_keep):
        with tf.variable_scope(self.name):
            return self._build(s_mixed, seq_lens, fw_drop_keep, recur_drop_keep)

    def _build(self, s_mixed, seq_lens, fw_drop_keep, recur_drop_keep):
        raise NotImplementedError()


class Estimator(object):
    def __init__(self, cfg, name=None):
        self._cfg = cfg
        self.name = name

    def gen_tf_mask(self, s_sources):
        # s_sources shape is [B, N, T, F]
        mask_type = self._cfg.mask_type
        N = self._cfg.MAX_SOURCE_NUM
        epsilon = 1e-7
        if mask_type == "irm":
            sum_magn = tf.reduce_sum(s_sources, axis=1, keepdims=True) + epsilon
            mask = s_sources / sum_magn
        elif mask_type == "ibm":
            arg_idx = tf.argmax(s_sources, axis=1)
            mask = tf.one_hot(arg_idx, depth=N, axis=1)
        elif mask_type == "wfm":
            sum_sqr = tf.reduce_sum(s_sources ** 2, axis=1, keepdims=True) + epsilon
            mask = s_sources ** 2 / sum_sqr
        else:
            raise ValueError("mask_type not supported: {}".format(mask_type))
        self.mask = tf.cast(mask, dtype=tf.float32)

    def estimate(self, s_embed, s_mixed, s_sources, seq_mask):
        self.gen_tf_mask(s_sources)
        with tf.variable_scope(self.name):
            return self._build(s_embed, s_mixed, s_sources, seq_mask)

    def _build(self, s_embed, s_mixed, s_sources, seq_mask):
        raise NotImplementedError()


class Separator(object):
    def __init__(self, cfg, name=None):
        self._cfg = cfg
        self.name = name

    def separate(self, s_mixed, s_attractors, s_embed):
        with tf.variable_scope(self.name):
            return self._build(s_mixed, s_attractors, s_embed)

    def _build(self, s_mixed, s_attractors, s_embed):
        raise NotImplementedError()


class LstmEncoder(Encoder):
    def __init__(self, cfg, name=None):
        super().__init__(cfg, name)
        if self.name is None:
            self.name = type(self).__name__

    def _build(self, s_mixed, seq_lens, fw_drop_keep, recur_drop_keep):
        feats_dim = self._cfg.feats_dim
        embedding_dim = self._cfg.embedding_dim
        project_size = feats_dim * embedding_dim
        encoder_dim = self._cfg.encoder_dim
        encoder_layers = self._cfg.encoder_layers
        bidirectional = self._cfg.bidirectional
        rnn = self._cfg.encoder_rnn_type
        if rnn == "rnn":
            rnn_cell = tf.contrib.rnn.BasicRNNCell
        elif rnn == "gru":
            rnn_cell = tf.contrib.rnn.GRUCell
        elif rnn == "basic_lstm":
            rnn_cell = tf.contrib.rnn.BasicLSTMCell
        elif rnn == "lstm":
            rnn_cell = tf.contrib.rnn.LSTMCell
        elif rnn == "layer_norm_lstm":
            rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell
        else:
            raise ValueError("encoder_rnn_type not supported {}".format(rnn))
        rnn_input = tf.identity(s_mixed)
        for i in range(encoder_layers):
            scope = "rnn_layer_%d" % (i+1)
            rnn_input, _ = rnn_layer(rnn_input, rnn_cell, encoder_dim, tf.nn.tanh,
                                     seq_lens, scope=scope, bidirectional=bidirectional,
                                     time_major=False, fw_dropout_keep=fw_drop_keep,
                                     recur_dropout_keep=recur_drop_keep)
        s_project = tf.layers.dense(rnn_input, project_size, activation=None, name='fc',
                                    use_bias=True, kernel_initializer=tf.truncated_normal_initializer(stddev=0.05))
        s_embed_out = tf.reshape(s_project, tf.concat([tf.shape(s_mixed), [embedding_dim]], 0))
        return s_embed_out


class AverageEstimator(Estimator):
    def __init__(self, cfg, name=None):
        super().__init__(cfg, name)
        if self.name is None:
            self.name = type(self).__name__

    def _build(self, s_embed, s_mixed, s_sources, seq_mask):
        embedding_dim = tf.shape(s_embed)[3]
        batch_size = tf.shape(s_embed)[0]
        N = self._cfg.MAX_SOURCE_NUM
        # [B, 1, T*F, 1]
        count = tf.multiply(tf.ones_like(s_mixed), tf.expand_dims(seq_mask, axis=2))
        count = tf.reshape(count, [batch_size, 1, -1, 1])
        # shape is [B, 1, T*F, E]
        s_embed_flat = tf.reshape(s_embed, [batch_size, 1, -1, embedding_dim])
        s_embed_flat = s_embed_flat * count
        # shape is [B, N, T*F, 1]
        tf_mask = tf.reshape(self.mask, [batch_size, N, -1, 1])
        tf_mask = tf_mask * count
        # [B, N, E]
        s_attractors = tf.reduce_sum(s_embed_flat * tf_mask, axis=2)
        s_attractors_wgt = tf.reduce_sum(tf_mask, axis=2) + 1  # avoid to divide 0
        s_attractors = s_attractors / s_attractors_wgt
        return s_attractors


class ThresholdedAverageEstimator(Estimator):
    def __init__(self, cfg, name=None):
        super().__init__(cfg, name)
        if self.name is None:
            self.name = type(self).__name__

    def _build(self, s_embed, s_mixed, s_sources, seq_mask):
        embedding_dim = tf.shape(s_embed)[3]
        batch_size = tf.shape(s_embed)[0]
        N = self._cfg.MAX_SOURCE_NUM
        threshold = self._cfg.threshold
        s_wgt = tf.cast(tf.less(threshold, s_mixed), tf.float32)
        s_wgt = tf.reshape(s_wgt, [batch_size, 1, -1, 1])
        # [B, 1, T*F, 1]
        count = tf.multiply(tf.ones_like(s_mixed), tf.expand_dims(seq_mask, axis=2))
        count = tf.reshape(count, [batch_size, 1, -1, 1])
        count = count * s_wgt
        # shape is [B, 1, T*F, E]
        s_embed_flat = tf.reshape(s_embed, [batch_size, 1, -1, embedding_dim])
        s_embed_flat = s_embed_flat * count
        # shape is [B, N, T*F, 1]
        tf_mask = tf.reshape(self.mask, [batch_size, N, -1, 1])
        tf_mask = tf_mask * count
        # [B, N, E]
        s_attractors = tf.reduce_sum(s_embed_flat * tf_mask, axis=2)
        s_attractors_wgt = tf.reduce_sum(tf_mask, axis=2) + 1  # avoid to divide 0
        s_attractors = s_attractors / s_attractors_wgt
        return s_attractors


class WeightedAverageEstimator(Estimator):
    def __init__(self, cfg, name=None):
        super().__init__(cfg, name)
        if self.name is None:
            self.name = type(self).__name__

    def _build(self, s_embed, s_mixed, s_sources, seq_mask):
        embedding_dim = tf.shape(s_embed)[3]
        batch_size = tf.shape(s_embed)[0]
        N = self._cfg.MAX_SOURCE_NUM
        s_wgt = tf.reshape(s_mixed, [batch_size, 1, -1, 1])
        # [B, 1, T*F, 1]
        count = tf.multiply(tf.ones_like(s_mixed), tf.expand_dims(seq_mask, axis=2))
        count = tf.reshape(count, [batch_size, 1, -1, 1])
        count = count * s_wgt
        # shape is [B, 1, T*F, E]
        s_embed_flat = tf.reshape(s_embed, [batch_size, 1, -1, embedding_dim])
        s_embed_flat = s_embed_flat * count
        # shape is [B, N, T*F, 1]
        tf_mask = tf.reshape(self.mask, [batch_size, N, -1, 1])
        tf_mask = tf_mask * count
        # [B, N, E]
        s_attractors = tf.reduce_sum(s_embed_flat * tf_mask, axis=2)
        s_attractors_wgt = tf.reduce_sum(tf_mask, axis=2) + 1  # avoid to divide 0
        s_attractors = s_attractors / s_attractors_wgt
        return s_attractors


class AnchoredEstimator(Estimator):
    def __init__(self, cfg, name=None):
        super().__init__(cfg, name)
        if self.name is None:
            self.name = type(self).__name__

    def _build(self, s_embed, s_mixed, s_sources, seq_mask):
        embedding_dim = self._cfg.embedding_dim
        batch_size = tf.shape(s_embed)[0]
        N = self._cfg.MAX_SOURCE_NUM
        num_anchors = self._cfg.num_anchors
        v_anchors = tf.get_variable("anchors", shape=[num_anchors, embedding_dim], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=1.0))
        # [P, N, E]
        s_anchor_sets = get_combinations(v_anchors, N)
        # [B, P, T, F, N]
        s_anchor_assignment = tf.einsum('btfe,pne->bptfn', s_embed, s_anchor_sets)
        s_anchor_assignment = tf.nn.softmax(s_anchor_assignment, axis=-1)
        s_anchor_assignment = tf.einsum('bptfn,bt->bptfn', s_anchor_assignment, seq_mask)
        # [B, P, N, E]
        s_attractor_sets = tf.einsum('bptfn,btfe->bpne', s_anchor_assignment, s_embed)
        s_attractor_sets /= tf.expand_dims(
            tf.reduce_sum(s_anchor_assignment, axis=(2, 3)), axis=-1)

        s_inset_affinity = tf.matmul(s_attractor_sets, tf.transpose(s_attractor_sets, [0, 1, 3, 2]))
        intra_index = get_intra_index(N)
        # [B, P]
        s_intra_similarity = tf.reduce_max(
            tf.gather_nd(tf.transpose(s_inset_affinity, [2, 3, 0, 1]), intra_index), axis=0
        )
        s_subset_choice = tf.cast(tf.argmin(s_intra_similarity, axis=1), dtype=tf.int32)
        s_subset_choice = tf.stack([tf.range(batch_size, dtype=tf.int32), s_subset_choice], axis=1)
        s_attractors = tf.gather_nd(s_attractor_sets, s_subset_choice)
        return s_attractors


class DotSeparator(Separator):
    def __init__(self, cfg, name=None):
        super().__init__(cfg, name)
        if self.name is None:
            self.name = type(self).__name__

    def _build(self, s_mixed, s_attractors, s_embed):
        feats_dim = self._cfg.feats_dim
        batch_size = tf.shape(s_attractors)[0]
        embedding_dim = self._cfg.embedding_dim
        N = self._cfg.MAX_SOURCE_NUM
        act_type = self._cfg.separator_activation
        s_embed_flat = tf.reshape(s_embed, [batch_size, -1, embedding_dim])

        s_logits = tf.matmul(s_embed_flat, tf.transpose(s_attractors, [0, 2, 1]))
        s_logits = tf.reshape(s_logits, [batch_size, -1, feats_dim, N])
        if act_type == "softmax":
            s_masks = tf.nn.softmax(s_logits, axis=-1)
        elif act_type == "sigmoid":
            s_masks = tf.nn.sigmoid(s_logits)
        else:
            raise ValueError("separator_activation not supported: {}".format(act_type))
        s_separated_log_magn = tf.expand_dims(s_mixed, axis=-1) * s_masks
        s_separated_log_magn = tf.transpose(s_separated_log_magn, [0, 3, 1, 2])
        return s_separated_log_magn


def rnn_layer(layer_input, rnn_cell, num_hidden, activation,
              seq_lens, scope, bidirectional=False, time_major=False,
              fw_dropout_keep=1.0, recur_dropout_keep=1.0):
    if bidirectional:
        fw_cell = rnn_cell(num_hidden, activation=activation,
                           reuse=tf.get_variable_scope().reuse)
        bw_cell = rnn_cell(num_hidden, activation=activation,
                           reuse=tf.get_variable_scope().reuse)
        initial_fw = fw_cell.zero_state(tf.shape(layer_input)[0], tf.float32)
        initial_bw = bw_cell.zero_state(tf.shape(layer_input)[0], tf.float32)
        fw_cell_dr = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=fw_dropout_keep,
                                                   state_keep_prob=recur_dropout_keep)
        bw_cell_dr = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=fw_dropout_keep,
                                                   state_keep_prob=recur_dropout_keep)
        layer, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell_dr, bw_cell_dr, layer_input,
                                                   sequence_length=tf.to_int32(seq_lens),
                                                   initial_state_fw=initial_fw,
                                                   initial_state_bw=initial_bw,
                                                   scope=scope, dtype=tf.float32,
                                                   time_major=time_major)
        layer = tf.concat(layer, axis=2)
        out_dim = num_hidden * 2
    else:
        fw_cell = rnn_cell(num_hidden, activation=activation,
                           reuse=tf.get_variable_scope().reuse)
        initial_fw = fw_cell.zero_state(tf.shape(layer_input)[0], tf.float32)
        fw_cell_dr = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=fw_dropout_keep,
                                                   state_keep_prob=recur_dropout_keep)
        layer, _ = tf.nn.dynamic_rnn(fw_cell_dr, layer_input,
                                     sequence_length=seq_lens,
                                     initial_state=initial_fw, scope=scope,
                                     dtype=tf.float32, time_major=time_major)
        out_dim = num_hidden
    return layer, out_dim


def get_seq_mask(s_mixed, seq_lens):
    max_len = tf.shape(s_mixed)[1]
    r = tf.range(max_len, dtype=tf.int32)
    func = lambda x: tf.cast(tf.less(r, x), dtype=tf.float32)
    mask = tf.map_fn(func, seq_lens, dtype=tf.float32)
    return mask


def get_intra_index(N):
    r = tf.range(N, dtype=tf.float32)
    def func(x):
        outer_axis = tf.ones([N - 1]) * x
        idx = tf.concat([tf.range(0, x), tf.range(x+1, N)], axis=0)
        idx = tf.stack([outer_axis, idx], axis=1)
        return idx
    intra_index = tf.to_int32(tf.map_fn(func, r))
    intra_index = tf.reshape(intra_index, [-1, 2])
    return intra_index


def get_combinations(s_data, sub_size, total_size=None):
    sub_size = int(sub_size)
    assert sub_size > 0
    if total_size is None:
        total_size = s_data.get_shape().as_list()[0]
    assert sub_size <= total_size
    c_comb = tf.constant(list(itertools.combinations(range(total_size), sub_size)),
                         dtype=tf.int32, name='c_comb')
    return tf.gather(s_data, c_comb)

