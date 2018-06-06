import os
import sys
lib_path = os.path.abspath('..')
sys.path.append(lib_path)

import logging
import time
import shutil
import numpy as np
import tensorflow as tf
from utils.signalprocess import *
from sklearn.cluster import KMeans

def ceil_divide(a, b):
    return (a + b - 1) // b


def set_log(log_dir):
    log_path = os.path.join(log_dir, 'log.out')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_format = ('%(asctime)s %(filename)s [line:%(lineno)d] '
                  '%(levelname)s %(message)s')
    logging.basicConfig(level=logging.DEBUG,
                        format=log_format,
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode='w')


def get_num_gpus(gpu):
    gpu = gpu.replace(' ', '')
    num_gpus = len(gpu.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return num_gpus


def create_folders(dir_path):
    os.makedirs(dir_path, exist_ok=True)


class MetricChecker(object):
    def __init__(self, cfg):
        self.learning_rate = cfg.learning_rate
        self.min_learning_rate = cfg.min_learning_rate
        self.decay_lr = cfg.decay_lr
        self.decay_lr_count = cfg.decay_lr_count
        self.early_stop_count = cfg.early_stop_count
        self.reset_step()
        self.curr_valid = tf.placeholder(tf.float32, shape=[], name="curr_valid")
        self.best_valid = tf.get_variable(name="best_valid", trainable=False, shape=[],
                                          initializer=tf.constant_initializer(np.inf))
        self.valid_imporved = tf.less(self.curr_valid, self.best_valid)
        with tf.control_dependencies([self.valid_imporved]):
            self.update_best_valid = tf.assign(self.best_valid,
                                               tf.minimum(self.best_valid, self.curr_valid))
    def update(self, sess, curr_valid):
        valid_improved, best_valid = sess.run([self.valid_imporved, self.update_best_valid],
                                              feed_dict={self.curr_valid : curr_valid})
        if valid_improved:
            self.reset_step()
        else:
            self.stop_step += 1
            self.lr_step += 1
            if self.lr_step == self.decay_lr_count:
                self.learning_rate = max(self.learning_rate * self.decay_lr, self.min_learning_rate)
                self.lr_step = 0
        return valid_improved, best_valid

    def reset_step(self):
        self.stop_step = 0
        self.lr_step = 0

    def should_stop(self):
        return self.stop_step >= self.early_stop_count

    def get_best(self, sess):
        return sess.run(self.best_valid)


def checker(cfg):
    with tf.variable_scope("ValidLossChecker"):
        valid_loss_checker = MetricChecker(cfg)
    return valid_loss_checker


class Prework(object):
    def __init__(self, cfg):
        self.job_dir = cfg.job_dir
        self.sorted_train_dir = cfg.train_dir
        self.sorted_valid_dir = cfg.valid_dir
        self.global_cmvn_norm = cfg.global_cmvn_norm
        self.global_cmvn_file = cfg.global_cmvn_file
        self.train_dir = os.path.join(self.job_dir, 'data', 'train')
        self.valid_dir = os.path.join(self.job_dir, 'data', 'valid')
        self.train_event_dir = os.path.join(self.job_dir, 'train_event_dir')
        self.valid_event_dir = os.path.join(self.job_dir, 'valid_event_dir')
        self.best_loss_dir = os.path.join(self.job_dir, 'best_loss')
        self.eval_addr = cfg.eval_dir
        self.eval_dir = os.path.join(self.job_dir, 'data', 'eval')
        self.eval_data_dir = os.path.join(self.eval_dir, cfg.eval_name)
        self.num_sources = cfg.MAX_SOURCE_NUM

    def copy_data_dir(self, src_dir, dst_dir):
        for file_name in ['feats.scp', 'gender.scp', 'feats.len']:
            src_file = os.path.join(src_dir, file_name)
            dst_file = os.path.join(dst_dir, file_name)
            if not os.path.exists(src_file):
                raise FileNotFoundError("{} not founded".format(src_file))
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_file)

    def make_env(self):
        create_folders(self.train_dir)
        create_folders(self.valid_dir)
        create_folders(self.train_event_dir)
        create_folders(self.valid_event_dir)
        create_folders(self.best_loss_dir)
        shutil.copy("config.py", os.path.join(self.job_dir, "config.py"))
        self.copy_data_dir(self.sorted_train_dir, self.train_dir)
        self.copy_data_dir(self.sorted_valid_dir, self.valid_dir)
        if self.global_cmvn_norm:
            dst_path = os.path.join(self.train_dir, 'global.cmvn')
            shutil.copy(self.global_cmvn_file, dst_path)
            self.global_cmvn_file = dst_path

    def make_eval_env(self):
        if self.global_cmvn_norm:
            cmvn_path = os.path.join(self.train_dir, 'global.cmvn')
            assert os.path.exists(cmvn_path)
            self.global_cmvn_file = cmvn_path
        create_folders(self.eval_dir)
        if os.path.exists(self.eval_data_dir):
            shutil.rmtree(self.eval_data_dir)
        for i in range(1, self.num_sources + 1):
            estimate_dir = os.path.join(self.eval_data_dir, "s%d"%i)
            create_folders(estimate_dir)
        feats_scp = os.path.join(self.eval_addr, "feats.scp")
        feats_len = os.path.join(self.eval_addr, "feats.len")
        shutil.copy(feats_scp, self.eval_data_dir)
        shutil.copy(feats_len, self.eval_data_dir)


def load_model(cfg, job_env, saver, sess):
    load_option = cfg.load_option
    # load from latest checkpoint
    if load_option == 0:
        load_path = tf.train.latest_checkpoint(checkpoint_dir=job_env.job_dir)
    # load from best loss dir
    elif load_option == 1:
        ckpt = tf.train.get_checkpoint_state(job_env.best_loss_dir)
        load_path = ckpt.model_checkpoint_path
    elif load_option == 2:
        load_path = cfg.load_path
    else:
        raise ValueError("it's not a supported load option")
    try:
        saver.restore(sess, load_path)
    except Exception as e:
        logging.error("Failed to load from {}".format(load_path))
        raise e
    return load_path


def create_valid_summary(valid_loss, learning_rate):
    values = [
        tf.Summary.Value(tag='valid_loss', simple_value=float(valid_loss)),
        tf.Summary.Value(tag='learning_rate', simple_value=float(learning_rate))
    ]
    summary = tf.Summary(value=values)
    return summary


def report_train(epoch, step, train_cursor, num_batches,
                 train_loss, iter_time, i_merge, train_summary_writer):
    msg = "Epoch = {} ({}/{}), Step = {}, Train_loss = {:6.4f}," \
          "Time = {:4.2f} sec"
    logging.info(msg.format(epoch + 1, train_cursor + 1, num_batches,
                            step, train_loss, iter_time))
    train_summary_writer.add_summary(i_merge, step)
    train_summary_writer.flush()


def report_valid(sess, step, handle, valid_iterator, num_batches, model,
                 saver, valid_checker, job_env, valid_summary_writer):
    valid_handle = sess.run(valid_iterator.string_handle())
    sess.run(valid_iterator.initializer)
    logging.info("Starting validation")
    all_loss = 0
    for i in range(num_batches):
        dev_time = time.time()
        d_loss = sess.run(model.loss,
                          feed_dict={handle: valid_handle,
                                     model.fw_dropout_keep: 1.0,
                                     model.recur_dropout_keep: 1.0})
        iter_time = time.time() - dev_time
        logging.info("ValidBatch {}/{}, valid_loss = {:6.4f}, "
                     "time = {:4.2f}".format(i + 1, num_batches, d_loss, iter_time))
        all_loss += d_loss
    avg_loss = all_loss / num_batches
    logging.info("AVG Valid loss = {:6.4f}".format(avg_loss))
    valid_improved, best_valid_loss = valid_checker.update(sess, avg_loss)
    if valid_improved:
        logging.info("New best valid loss: {:6.4f}".format(best_valid_loss))
        saver.save(sess, os.path.join(job_env.best_loss_dir, 'model.ckpt'))
    valid_summary = create_valid_summary(avg_loss, valid_checker.learning_rate)
    valid_summary_writer.add_summary(valid_summary, step)
    valid_summary_writer.flush()


def sigmoid(x):
    e_negx = np.exp(-x)
    return 1.0 / (1 + e_negx)


def softmax(x, axis=-1):
    e_x = np.exp(x)
    sum_ex = np.sum(e_x, axis=axis, keepdims=True)
    return e_x / sum_ex


def separate(s_mixed, s_attractor, s_embed, act_type='softmax'):
    feats_dim = s_mixed.shape[2]
    batch_size = s_mixed.shape[0]
    N = s_attractor.shape[1]
    E = s_attractor.shape[2]
    s_embed_flat = np.reshape(s_embed, [batch_size, -1, E])
    s_cor = np.matmul(s_embed_flat, np.transpose(s_attractor, [0, 2, 1]))
    s_cor = np.reshape(s_cor, [batch_size, -1, feats_dim, N])
    if act_type == "softmax":
        s_mask = softmax(s_cor, axis=-1)
    elif act_type == "sigmoid":
        s_mask = sigmoid(s_cor)
    else:
        raise ValueError("not supported act_type: {}".format(act_type))
    s_logits = np.expand_dims(s_mixed, axis=-1) * s_mask
    s_logits = np.transpose(s_logits, [0, 3, 1, 2])
    return s_logits


def get_eval_estimated(sess, model, num_sources, input_style=0,
                       input_feats=None, seq_lens=None, act_type=None,
                       eval_type="anchor", feed_attractor=None):
    feed_dict = {model.fw_dropout_keep: 1.0, model.recur_dropout_keep: 1.0}
    if input_style == 1:
        assert input_feats is not None and seq_lens is not None
        feed_dict[model.group_input.feats] = input_feats
        feed_dict[model.group_input.seq_lens] = seq_lens
        if eval_type == "anchor":
            eval_logits = sess.run(model.tower_logits, feed_dict=feed_dict)
            return eval_logits[0]
        elif eval_type == "kmeans":
            assert act_type is not None
            feats_dim = input_feats.shape[-1] // 2
            mixed_magn = input_feats[:, :, 0:feats_dim]
            eval_embed = sess.run(model.tower_embed, feed_dict=feed_dict)
            eval_embed = eval_embed[0]
            embeding_dims = eval_embed.shape[3]
            attractors = []
            for i in range(eval_embed.shape[0]):
                embed_flat = np.reshape(eval_embed[i][0:seq_lens[i]], [-1, embeding_dims])
                kmeans = KMeans(n_clusters=num_sources, n_init=5,
                                max_iter=20).fit(embed_flat)
                attractors.append(kmeans.cluster_centers_)
            attractors = np.array(attractors)
            eval_logits = separate(mixed_magn, attractors, eval_embed,
                                   act_type=act_type)
            return eval_logits
        else:
            assert act_type is not None and feed_attractor is not None
            assert feed_attractor.ndims == 2
            assert feed_attractor.shape[0] == num_sources
            batch_size = input_feats.shape[0]
            feats_dim = input_feats.shape[-1] // 2
            mixed_magn = input_feats[:, :, 0:feats_dim]
            eval_embed = sess.run(model.tower_embed, feed_dict=feed_dict)
            eval_embed = eval_embed[0]
            assert feed_attractor.shape[1] == eval_embed.shape[3]
            tile_attractor = np.tile(feed_attractor, (batch_size, 1, 1))
            eval_logits = separate(mixed_magn, tile_attractor, eval_embed,
                                   act_type=act_type)
            return eval_logits
    elif input_style == 0:
        if eval_type == "anchor":
            fetch = [model.tower_logits, model.tower_mix_phase, model.tower_seq_lens]
            eval_logits, mixed_phase, seq_lens = sess.run(fetch, feed_dict=feed_dict)
            return eval_logits[0], mixed_phase[0], seq_lens[0]
        elif eval_type == "kmeans":
            assert act_type is not None
            fetch = [model.tower_embed, model.tower_mix_magn,
                     model.tower_mix_phase, model.tower_seq_lens]
            eval_embed, mixed_magn, mixed_phase, seq_lens = sess.run(fetch, feed_dict=feed_dict)
            eval_embed, mixed_magn, mixed_phase, seq_lens = eval_embed[0], mixed_magn[0], \
                                                            mixed_phase[0], seq_lens[0]
            embeding_dims = eval_embed.shape[3]
            attractors = []
            for i in range(eval_embed.shape[0]):
                embed_flat = np.reshape(eval_embed[i][0:seq_lens[i]], [-1, embeding_dims])
                kmeans = KMeans(n_clusters=num_sources, n_init=5,
                                max_iter=20).fit(embed_flat)
                attractors.append(kmeans.cluster_centers_)
            attractors = np.array(attractors)
            eval_logits = separate(mixed_magn, attractors, eval_embed,
                                   act_type=act_type)
            return eval_logits, mixed_phase, seq_lens
        else:
            assert act_type is not None and feed_attractor is not None
            assert feed_attractor.ndims == 2
            assert feed_attractor.shape[0] == num_sources
            fetch = [model.tower_embed, model.tower_mix_magn,
                     model.tower_mix_phase, model.tower_seq_lens]
            eval_embed, mixed_magn, mixed_phase, seq_lens = sess.run(fetch, feed_dict=feed_dict)
            eval_embed, mixed_magn, mixed_phase, seq_lens = eval_embed[0], mixed_magn[0], \
                                                            mixed_phase[0], seq_lens[0]
            batch_size = eval_embed.shape[0]
            assert feed_attractor.shape[1] == eval_embed.shape[3]
            tile_attractor = np.tile(feed_attractor, (batch_size, 1, 1))
            eval_logits = separate(mixed_magn, tile_attractor, eval_embed,
                                   act_type=act_type)
            return eval_logits, mixed_phase, seq_lens
    else:
        raise ValueError("not supported input_style: {}".format(input_style))


