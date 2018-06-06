import config as cfg
from utils.tools import *
from model.build_model import build_model

import os
import logging
import time
import tensorflow as tf
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


class Trainer(object):
    def __init__(self):
        pass

    def train(self, sess, num_gpus):
        tf.set_random_seed(cfg.seed)
        set_log(cfg.job_dir)
        job_env = Prework(cfg)
        job_env.make_env()

        assert cfg.input_style == 0
        model, handlers = build_model(cfg, job_env, num_gpus)
        handle, train_iterator, valid_iterator, train_num, valid_num = handlers
        valid_loss_checker = checker(cfg)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, name='train_saver')
        best_saver = tf.train.Saver(tf.global_variables(), name="best_saver")
        checkpoint_path = os.path.join(job_env.job_dir, 'model.ckpt')
        train_summary_writer = tf.summary.FileWriter(job_env.train_event_dir)
        valid_summary_writer = tf.summary.FileWriter(job_env.valid_event_dir)

        train_handle = sess.run(train_iterator.string_handle())

        if cfg.resume:
            load_path = load_model(cfg, job_env, saver, sess)
            logging.info("Loading the existing model: %s" % load_path)

        epoch, train_cursor = 0, 0
        group_size = num_gpus * cfg.batch_size
        train_num_batches = train_num // group_size
        valid_num_batches = valid_num // group_size
        while True:
            try:
                start_time = time.time()
                loss, _, i_global, i_merge = sess.run(
                    [model.loss, model.train_op, model.global_step, model.merged],
                    feed_dict={handle: train_handle,
                               model.learning_rate: valid_loss_checker.learning_rate,
                               model.fw_dropout_keep: cfg.fw_dropout_keep,
                               model.recur_dropout_keep: cfg.recur_dropout_keep})
                iter_time = time.time() - start_time
                if i_global % cfg.train_log_freq == 0:
                    report_train(epoch, i_global, train_cursor, train_num_batches,
                                 loss, iter_time, i_merge, train_summary_writer)
                if i_global % cfg.valid_freq == 0:
                    report_valid(sess, i_global, handle, valid_iterator, valid_num_batches,
                                 model, best_saver, valid_loss_checker, job_env, valid_summary_writer)
                    if valid_loss_checker.should_stop():
                        break
                if i_global % cfg.save_freq == 0:
                    saver.save(sess, checkpoint_path, global_step=i_global)
                train_cursor += 1
                if train_cursor == train_num_batches:
                    train_cursor = 0
                    epoch += 1
            except tf.errors.OutOfRangeError:
                break
        sess.close()


if __name__ == "__main__":
    num_gpus = get_num_gpus(cfg.gpu)
    print("Use {:d} gpus: {}".format(num_gpus, cfg.gpu))

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    sess = tf.Session(config=sess_config)
    trainer = Trainer()
    try:
        trainer.train(sess, num_gpus)
    except Exception as e:
        print('failed....')
        raise e


