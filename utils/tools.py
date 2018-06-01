import os
import logging
import time
import shutil
import numpy as np
import tensorflow as tf


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


def create_folders(dir_path):
    os.makedirs(dir_path, exist_ok=True)


class MetricChecker(object):
    def __init__(self, cfg):
        self.learning_rate = cfg.learning_rate
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
                self.learning_rate = self.learning_rate * self.decay_lr
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
        saver.load(sess, load_path)
    except Exception as e:
        logging.error("Failed to load from {}".format(load_path))
        raise e
    return load_path


def create_valid_summary(valid_loss):
    values = [
        tf.Summary.Value(tag='valid_loss', simple_value=float(valid_loss))
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
    valid_summary = create_valid_summary(avg_loss)
    valid_summary_writer.add_summary(valid_summary, step)
    valid_summary_writer.flush()
