import config as cfg
from utils.tools import *
from model.model_input import TF_EvalDataset
from model.build_model import build_eval_model
from utils.signalprocess import *

import os
import logging
import time
import shutil
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.logging.set_verbosity(tf.logging.ERROR)


class Evaluator(object):
    def __init__(self):
        pass

    def eval(self, sess, num_gpus):
        job_env = Prework(cfg)
        job_env.make_eval_env()
        set_log(job_env.eval_data_dir)
        dataset_input = TF_EvalDataset(data_dir=job_env.eval_data_dir, batch_size=cfg.batch_size,
                                       num_sources=cfg.MAX_SOURCE_NUM)
        key_list = dataset_input.key_list
        eval_num = len(key_list)
        next_batch = dataset_input.build(cfg.feats_dim * 2)

        # use one gpu to eval by default, avoiding to encounter the
        # problem that num_gpus does not evenly divide the num_sample in group
        model = build_eval_model(cfg, job_env, input_style=0, num_gpus=1, next_batch=next_batch)
        saver = tf.train.Saver(tf.trainable_variables())
        load_path = load_model(cfg, job_env, saver, sess)
        logging.info("Loading the existing model: %s" % load_path)

        num_sources = cfg.MAX_SOURCE_NUM
        msg = "Batch {}/{}, num_samples: {:>3d}, time: {:.2f} sec"
        logging.info("start evaluating")
        start_time = time.time()
        num_batches = ceil_divide(eval_num, cfg.batch_size)
        infer_duration, sample_count = 0, 0
        for i in range(num_batches):
            iter_time = time.time()
            batch_logits, batch_mix_phase, seq_lens = get_eval_estimated(sess, model, num_sources,
                                                                         input_style=0,
                                                                         eval_type=cfg.eval_estimator_type,
                                                                         act_type=cfg.separator_activation)
            iter_time = time.time() - iter_time
            num_sent = batch_logits.shape[0]
            for sent_i in range(num_sent):
                key_name = key_list[sample_count + sent_i]
                for n_i in range(batch_logits.shape[1]):
                    real = batch_logits[sent_i][n_i][0:seq_lens[sent_i]] * \
                        np.cos(batch_mix_phase[sent_i][0:seq_lens[sent_i]])
                    img = batch_logits[sent_i][n_i][0:seq_lens[sent_i]] * \
                        np.sin(batch_mix_phase[sent_i][0:seq_lens[sent_i]])
                    stft_data = real + 1j * img
                    recon = istft(stft_data, size=cfg.frame_size, shift=cfg.shift, fading=True)
                    wavpath = os.path.join(job_env.eval_data_dir, 's%d'%(n_i + 1), key_name + '.wav')
                    audiowrite(wavpath, recon, samp_rate=8000)
            sample_count += num_sent
            logging.info(msg.format(i + 1, num_batches, num_sent, iter_time))
            infer_duration += iter_time
        eval_duration = time.time() - start_time
        logging.info("Separating %d samples took %.2f seconds, infer_time %.2f%%" % (
            sample_count, eval_duration, infer_duration * 100 / eval_duration))
        sess.close()

    def get_feed_model(self, sess, num_gpus):
        # adjust pipeline with placeholder and load model to separate a given mixed wav
        job_env = Prework(cfg)
        if cfg.global_cmvn_norm:
            job_env.global_cmvn_file = os.path.join(job_env.train_dir, 'global.cmvn')
        model = build_eval_model(cfg, job_env, input_style=1, num_gpus=1)
        saver = tf.train.Saver(tf.trainable_variables())
        load_path = load_model(cfg, job_env, saver, sess)
        print("Loading the existing model: {}".format(load_path))
        return model, job_env

    def separate_sample(self, sess, model, job_env, sample_path):
        demo_dir = os.path.join(job_env.job_dir, 'data', 'eval', 'demo')
        os.makedirs(demo_dir, exist_ok=True)
        num_sources = cfg.MAX_SOURCE_NUM
        os.makedirs(os.path.join(demo_dir, 'mix'), exist_ok=True)
        for i in range(1, num_sources + 1):
            os.makedirs(os.path.join(demo_dir, 's%d'%i), exist_ok=True)
        base_name = os.path.basename(sample_path)
        shutil.copy(sample_path, os.path.join(demo_dir, 'mix', base_name))
        time_signal = audioread(sample_path, samp_rate=8000)
        stft_signal = stft(time_signal, size=cfg.frame_size, shift=cfg.shift)
        stft_magn = np.abs(stft_signal)
        stft_phase = np.angle(stft_signal)
        seq_lens = np.expand_dims(stft_magn.shape[0], axis=0)
        input_feats = np.expand_dims(np.concatenate([stft_magn, stft_phase], axis=1), axis=0)
        eval_logits = get_eval_estimated(sess, model, num_sources, input_style=1,
                                         input_feats=input_feats, seq_lens=seq_lens,
                                         eval_type=cfg.eval_estimator_type,
                                         act_type=cfg.separator_activation)
        eval_logits = eval_logits[0]
        for n_i in range(num_sources):
            real = eval_logits[n_i] * np.cos(stft_phase)
            img = eval_logits[n_i] * np.sin(stft_phase)
            stft_data = real + 1j * img
            recon = istft(stft_data, size=cfg.frame_size, shift=cfg.shift)
            wavpath = os.path.join(demo_dir, 's%d'%(n_i + 1), base_name)
            audiowrite(wavpath, recon, samp_rate=8000)


if __name__ == "__main__":
    num_gpus = get_num_gpus(cfg.gpu)
    print("Use {:d} gpus: {}".format(num_gpus, cfg.gpu))
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    sess = tf.Session(config=sess_config)
    evaluator = Evaluator()
    try:
        if cfg.job_type == "eval":
            # use one gpu in eval by default
            evaluator.eval(sess, 1)
        elif cfg.job_type == "demo":
            feed_model, job_env = evaluator.get_feed_model(sess, 1)
            evaluator.separate_sample(sess, feed_model, job_env, cfg.test_wav)
        else:
            raise ValueError("Not supported job_type: {}".format(cfg.job_type))
    except Exception as e:
        print("failed...")
        raise e
