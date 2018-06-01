import os, sys
libpath = os.path.abspath('..')
sys.path.append(libpath)

import random
import tensorflow as tf
from utils import kaldi_io

class TF_DatasetInput(object):
    def __init__(self, train_dir, num_epochs=10, batch_size=16,
                 start_shuffle_epoch=1, num_sources=2, valid_dir=None):
        self.epoch = 0
        self.num_epochs = num_epochs
        self.start_shuffle_epoch = start_shuffle_epoch
        self.batch_size = batch_size
        self.num_sources = num_sources
        self.train_list = self.get_list(train_dir)
        self.train_num = len(self.train_list)
        self.valid_list = None
        self.valid_num = None
        if valid_dir is not None:
            self.valid_list = self.get_list(valid_dir)
            self.valid_num = len(self.valid_list)
        if start_shuffle_epoch == 0:
            self.shuffle()

    def get_list(self, data_dir):
        feats_scp = os.path.join(data_dir, 'feats.scp')
        gender_scp = os.path.join(data_dir, 'gender.scp')
        len_scp = os.path.join(data_dir, 'feats.len')
        res_list = []
        with open(feats_scp, 'r') as ff, open(gender_scp, 'r') as gf, \
            open(len_scp, 'r') as lf:
            for f_line, g_line, l_line in zip(ff.readlines(), gf.readlines(), lf.readlines()):
                f_line = f_line.strip().split()
                g_line = g_line.strip().split()
                l_line = l_line.strip().split()
                assert f_line[0] == g_line[0] and f_line[0] == l_line[0]
                res_list.append((f_line[1], g_line[1], int(l_line[1])))
        return res_list

    def shuffle(self):
        batch = self.batch_size
        num_train = len(self.train_list)
        chunks = [self.train_list[i:i+batch] for i in range(0, num_train, batch)]
        random.shuffle(chunks)
        self.train_list = []
        for chunk in chunks:
            self.train_list.extend(chunk)

    def gen_train(self):
        for ele in self.train_list:
            feats = kaldi_io.read_mat(ele[0])
            gender = kaldi_io.read_vec_int(ele[1])
            yield feats, gender
        self.epoch += 1
        if self.epoch >= self.start_shuffle_epoch:
            self.shuffle()

    def gen_valid(self):
        for ele in self.valid_list:
            feats = kaldi_io.read_mat(ele[0])
            gender = kaldi_io.read_vec_int(ele[1])
            yield feats, gender

    def slice_example(self, feats, gender):
        frames = tf.cast(tf.shape(feats)[0] / 3, tf.int32)
        input_mix = feats[0:frames]
        # labels shape is [num_sources, frames, fft_size]
        labels = tf.reshape(tf.slice(feats, [frames, 0], [-1, -1]),
                            [self.num_sources, frames, -1])
        return input_mix, labels, frames, gender

    def build(self, fft_size):
        padded_shapes = (tf.TensorShape([None, fft_size]),
                         tf.TensorShape([self.num_sources, None, fft_size]),
                         tf.TensorShape([]),
                         tf.TensorShape([self.num_sources]))
        train_dataset = tf.data.Dataset.from_generator(self.gen_train, (tf.float32, tf.int32),
                                                       (tf.TensorShape([None, fft_size]), tf.TensorShape([self.num_sources])))
        train_dataset = train_dataset.map(self.slice_example).repeat(self.num_epochs)
        train_dataset = train_dataset.padded_batch(self.batch_size, padded_shapes=padded_shapes)
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes
        )
        next_batch = iterator.get_next()
        train_iterator = train_dataset.make_one_shot_iterator()
        if self.valid_list is not None:
            valid_dataset = tf.data.Dataset.from_generator(self.gen_valid, (tf.float32, tf.int32),
                                                           (tf.TensorShape([None, fft_size]), tf.TensorShape([self.num_sources])))
            valid_dataset = valid_dataset.map(self.slice_example)
            valid_dataset = valid_dataset.padded_batch(self.batch_size, padded_shapes=padded_shapes)
            valid_iterator = valid_dataset.make_initializable_iterator()
        else:
            valid_iterator = None
        return next_batch, handle, train_iterator, valid_iterator

def Create_Input_Placeholder(feat_dims, num_sources):
    input_mix = tf.placeholder(tf.float32, shape=[None, None, feat_dims], name='input')
    labels = tf.placeholder(tf.float32, shape=[None, num_sources, None, feat_dims], name='labels')
    seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')
    gender = tf.placeholder(tf.int32, shape=[None, num_sources], name='gender')
    return input_mix, labels, seq_len, gender

