import os
import sys
lib_path = os.path.abspath('..')
sys.path.append(lib_path)

import copy
import itertools
import numpy as np
import tensorflow as tf
from model.modules import *


class DANmodel(object):
    def __init__(self, cfg, batch_input, fw_dropout_keep,
                 recur_dropout_keep, cmvn=None):
        self._cfg = cfg
        self.cmvn = cmvn
        # shape is [B, T, F]
        s_mixed_magnitude, s_mixed_phase = tf.split(batch_input.feats, 2, axis=2)
        # shape is [B, N_SOURCE, T, F]
        s_source_magnitude, s_source_phase = tf.split(batch_input.labels, 2, axis=3)
        # log_e(1+x)
        # s_mixed_log_magn = tf.log1p(s_mixed_magnitude)
        # s_source_log_magn = tf.log1p(s_source_magnitude)
        seq_lens = batch_input.seq_lens
        genders = batch_input.gender
        self.s_mixed_magn = s_mixed_magnitude
        self.s_mixed_phase = s_mixed_phase
        self.seq_lens = seq_lens
        with tf.variable_scope("DANmodel"):
            self.build(s_mixed_magnitude, s_source_magnitude,
                       s_mixed_phase, s_source_phase, seq_lens,
                       genders, fw_dropout_keep, recur_dropout_keep)

    def build(self, s_mixed_magnitude, s_source_magnitude,
              s_mixed_phase, s_source_phase, seq_lens,
              genders, fw_dropout_keep, recur_dropout_keep):
        # log_e(1+x)
        s_mixed_log_magn = tf.log1p(s_mixed_magnitude)
        seq_mask = get_seq_mask(s_mixed_log_magn, seq_lens)
        batch_size = tf.shape(s_mixed_log_magn)[0]
        N = self._cfg.MAX_SOURCE_NUM
        embedding_dim = self._cfg.embedding_dim
        feats_dim = self._cfg.feats_dim

        # map input to embedding
        encoder = LstmEncoder(self._cfg, 'LstmEncoder')
        encoder_input = s_mixed_log_magn
        if self.cmvn is not None:
            mean_var = self.cmvn[:, 0]
            variance_var = self.cmvn[:, 1]
            encoder_input = (s_mixed_log_magn - mean_var) / variance_var
        self.s_embed = encoder.encode(encoder_input, seq_lens, fw_dropout_keep, recur_dropout_keep)

        # estimator in train and valid process
        estimator_type = self._cfg.estimator_type
        if estimator_type == "avg":
            estimator = AverageEstimator(self._cfg, 'AverageEstimator')
        elif estimator_type == "avg_thresh":
            estimator = ThresholdedAverageEstimator(self._cfg, 'ThreshAVGEstimator')
        elif estimator_type == "avg_weighted":
            estimator = WeightedAverageEstimator(self._cfg, 'WeightedAVGEstimator')
        elif estimator_type == "anchor":
            estimator = AnchoredEstimator(self._cfg, 'AnchoredEstimator')
        else:
            raise ValueError("estimator_type not supported: {}".format(estimator_type))
        self.attractors = estimator.estimate(self.s_embed, s_mixed_magnitude, s_source_magnitude, seq_mask)

        # separator
        separator = DotSeparator(self._cfg)
        self.not_align_separated_logits = separator.separate(s_mixed_magnitude, self.attractors, self.s_embed)


        # [B, N, T, F]
        # TODO: involve mix and source phase to implement Phase sensitive mask(complex loss can deal),
        #       softmax act correspond IRM, sigmoid act correspond IAM
        if self._cfg.loss_phase_sense:
            expand_mix_phase = tf.expand_dims(s_mixed_phase, axis=1)
            separated_logits = tf.complex(
                tf.cos(expand_mix_phase) * self.not_align_separated_logits,
                tf.sin(expand_mix_phase) * self.not_align_separated_logits)
            source_complex = tf.complex(
                tf.cos(s_source_phase) * s_source_magnitude,
                tf.sin(s_source_phase) * s_source_magnitude)
            min_loss, v_perms, min_loss_idx = self.pit_mse_loss(separated_logits,
                                                                source_complex, seq_mask)
        else:
            min_loss, v_perms, min_loss_idx = self.pit_mse_loss(self.not_align_separated_logits,
                                                                s_source_magnitude, seq_mask)
        self.loss = tf.reduce_mean(min_loss)

        train_perm_idxs = tf.stack([
            tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, N]),
            tf.gather(v_perms, min_loss_idx)], axis=2)
        train_perm_idxs = tf.reshape(train_perm_idxs, [-1, 2])
        align_separated_logits = tf.gather_nd(self.not_align_separated_logits, train_perm_idxs)
        self.align_separated_logits = tf.reshape(align_separated_logits, [batch_size, N, -1, feats_dim])


    def pit_mse_loss(self, s_x, s_y, seq_mask, name='pit_loss'):
        batch_size = tf.shape(s_x)[0]
        N = self._cfg.MAX_SOURCE_NUM
        expand_mask = tf.reshape(seq_mask, [batch_size, 1, 1, -1, 1])
        tf_bins_count = tf.reshape(tf.reduce_sum(seq_mask, axis=1) * self._cfg.feats_dim,
                                   [batch_size, 1, 1])
        with tf.variable_scope(name):
            v_perms = tf.constant(
                list(itertools.permutations(range(N))),
                dtype=tf.int32
            )
            perms_one_hot = tf.one_hot(v_perms, depth=N, dtype=tf.float32)
            actual_x = tf.expand_dims(s_x, 1)
            actual_y = tf.expand_dims(s_y, 2)
            if s_x.dtype.is_complex and s_y.dtype.is_complex:
                s_diff = actual_x - actual_y
                pairwise_loss = tf.reduce_sum(
                    tf.square(tf.real(s_diff) * expand_mask) +
                    tf.square(tf.imag(s_diff) * expand_mask),
                    axis=(3, 4)) / tf_bins_count
            else:
                actual_x = actual_x * expand_mask
                actual_y = actual_y * expand_mask
                pairwise_loss = tf.reduce_sum(
                    tf.squared_difference(actual_x, actual_y),
                    axis=(3, 4)) / tf_bins_count
            # [B, P]
            loss_set = tf.einsum('bij,pij->bp', pairwise_loss, perms_one_hot)
            min_loss_idx = tf.cast(tf.argmin(loss_set, axis=1), dtype=tf.int32)
            min_loss = tf.gather_nd(
                loss_set,
                tf.stack([tf.range(batch_size, dtype=tf.int32), min_loss_idx], axis=1)
            )
        return min_loss, v_perms, min_loss_idx

    def get_mix_category(self, genders):
        has_men = tf.map_fn(lambda x: tf.to_int32(tf.reduce_any(tf.equal(x, 1))), genders)
        has_women = tf.map_fn(lambda x: tf.to_int32(tf.reduce_any(tf.equal(x, 0))), genders)
        # 2: only man;  1: both sexes ; 0: only woman
        category = has_men + 1 - has_women
        return tf.to_int32(category)

    def get_loss(self):
        return self.loss

    def get_logits(self):
        return self.not_align_separated_logits
