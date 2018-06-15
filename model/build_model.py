import os
import sys
lib_path = os.path.abspath('..')
sys.path.append(lib_path)

import time
import re
import numpy as np
import tensorflow as tf
from model.model_input import TF_DatasetInput, Create_Input_Placeholder
from model.DANmodel import DANmodel
from collections import namedtuple

Separation_Input = namedtuple('Separation_Input', ['feats', 'labels', 'seq_lens', 'gender'])
Dataset_handlers = namedtuple('Dataset_handlers', ['handle', 'train_iter', 'valid_iter', 'train_num', 'valid_num'])

class Model(object):
    """
    this is wrapper of multiple gpu style
    The input for different gpu is simply divided from a group input, thus num_sample in group input
    should be evenly divided by num_gpus. Also, the train_num and valid_num should be evenly divided
    by group_size
    """
    def __init__(self, cfg, num_gpus, group_input, cmvn=None):
        self.num_gpus = num_gpus
        self.group_input = group_input
        self.global_step = tf.get_variable(name='global_step', shape=[], trainable=False,
                                           initializer=tf.constant_initializer(0), dtype=tf.int32)
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.fw_dropout_keep = tf.placeholder(tf.float32, shape=[], name="fw_dropout_keep")
        self.recur_dropout_keep = tf.placeholder(tf.float32, shape=[], name="recur_dropout_keep")
        self.cmvn = cmvn
        self.multi_gpu_model(cfg, num_gpus, group_input, self.global_step)

    def multi_gpu_model(self, cfg, num_gpus, group_input, global_step):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        tower_grads = []
        tower_loss = []
        self.tower_logits = []
        self.tower_mix_phase = []
        self.tower_seq_lens = []
        self.tower_mix_magn = []
        self.tower_embed = []
        self.tower_attractor = []
        feats_lst, labels_lst, seq_lens_lst, genders_lst = self.split_group(group_input, num_gpus)
        for i in range(num_gpus):
            worker = '/gpu:%d' % i
            device_setter = tf.train.replica_device_setter(
                worker_device=worker, ps_device='/cpu:0', ps_tasks=1)
            reuse = bool(i != 0)
            with tf.variable_scope("multi_gpu", reuse=reuse):
                with tf.name_scope('tower_%d' % i):
                    with tf.device(device_setter):
                        batch_input = Separation_Input(feats_lst[i], labels_lst[i],
                                                       seq_lens_lst[i], genders_lst[i])
                        model = DANmodel(cfg, batch_input, self.fw_dropout_keep,
                                         self.recur_dropout_keep, self.cmvn)
                        loss = model.get_loss()
                        weight_decay = cfg.weight_decay
                        if weight_decay > 0.0:
                            exclude_pattern = cfg.weight_decay_exclude
                            for param in regularizable_variables(exclude_pattern):
                                loss += weight_decay * tf.nn.l2_loss(param)

                        logits = model.get_logits()
                        grad_var = optimizer.compute_gradients(loss)
                        tower_grads.append(grad_var)
                        tower_loss.append(loss)
                        self.tower_logits.append(logits)
                        self.tower_mix_phase.append(model.s_mixed_phase)
                        self.tower_seq_lens.append(model.seq_lens)
                        self.tower_mix_magn.append(model.s_mixed_magn)
                        self.tower_attractor.append(model.attractors)
                        self.tower_embed.append(model.s_embed)
        avg_grad_var = average_gradients(tower_grads, cfg.clip_grad)
        self.train_op = optimizer.apply_gradients(avg_grad_var, global_step=global_step)
        self.loss = tf.reduce_mean(tower_loss)
        tf.summary.scalar("train_loss", self.loss)
        self.merged = tf.summary.merge_all()

    def split_group(self, group_input, num_gpus):
        feats, labels, seq_lens, genders = group_input
        feats_lst = tf.split(feats, num_gpus, axis=0)
        labels_lst = tf.split(labels, num_gpus, axis=0)
        seq_lens_lst = tf.split(seq_lens, num_gpus, axis=0)
        genders_lst = tf.split(genders, num_gpus, axis=0)
        return feats_lst, labels_lst, seq_lens_lst, genders_lst

    def set_global_step(self, sess, step):
        sess.run(tf.assign(self.global_step, step))

    def get_global_step(self, sess):
        return sess.run(self.global_step)


def build_model(cfg, job_env, num_gpus=1):
    input_style = cfg.input_style
    if cfg.global_cmvn_norm:
        cmvn = np.loadtxt(job_env.global_cmvn_file, dtype=np.float32)
        # get theta
        cmvn[:, 1] = np.sqrt(cmvn[:, 1])
    else:
        cmvn = None
    if input_style == 0:  # use TF_dataset as input pipeline
        group_size = cfg.batch_size * num_gpus
        dataset_input = TF_DatasetInput(train_dir=job_env.train_dir, num_epochs=cfg.MAX_EPOCHS,
                                        batch_size=group_size, start_shuffle_epoch=cfg.start_shuffle_epoch,
                                        num_sources=cfg.MAX_SOURCE_NUM, valid_dir=job_env.valid_dir)
        next_batch, handle, train_iterator, valid_iterator = dataset_input.build(cfg.feats_dim * 2)
        group_input = Separation_Input(next_batch[0], next_batch[1],
                                       next_batch[2], next_batch[3])
        model = Model(cfg, num_gpus, group_input, cmvn=cmvn)
        handlers = Dataset_handlers(handle, train_iterator, valid_iterator,
                                    dataset_input.train_num, dataset_input.valid_num)
        return model, handlers
    elif input_style == 1:  # use placeholder as input
        feats, labels, seq_lens, genders = Create_Input_Placeholder(cfg.feats_dim * 2,
                                                                    cfg.MAX_SOURCE_NUM)
        group_input = Separation_Input(feats, labels, seq_lens, genders)
        model = Model(cfg, num_gpus, group_input, cmvn=cmvn)
        return model
    else:
        raise NotImplementedError


def build_eval_model(cfg, job_env, input_style, num_gpus=1, next_batch=None):
    if cfg.global_cmvn_norm:
        cmvn = np.loadtxt(job_env.global_cmvn_file, dtype=np.float32)
    else:
        cmvn = None
    if input_style == 0:
        assert next_batch is not None
        group_input = Separation_Input(next_batch[0], next_batch[1],
                                       next_batch[2], next_batch[3])
        model = Model(cfg, num_gpus, group_input, cmvn=cmvn)
    elif input_style == 1:
        feats, labels, seq_lens, genders = Create_Input_Placeholder(cfg.feats_dim * 2,
                                                                    cfg.MAX_SOURCE_NUM)
        group_input = Separation_Input(feats, labels, seq_lens, genders)
        model = Model(cfg, num_gpus, group_input, cmvn=cmvn)
    else:
        raise NotImplementedError
    return model


def average_gradients(tower_grads, clip_grad):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expand_g = tf.expand_dims(g, 0)
            grads.append(expand_g)
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, 0)
        grad = tf.clip_by_norm(grad, clip_grad)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def regularizable_variables(exclude_pattern):
    pattern = re.compile(exclude_pattern)
    output = [param for param in tf.trainable_variables()
              if not pattern.search(param.name)]
    return output
