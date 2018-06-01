import os, sys
lib_path = os.path.abspath('..')
sys.path.append(lib_path)

import random
import tensorflow as tf
import numpy as np
from utils import kaldi_io

os.environ['CUDA_VISIBLE_DEVICES'] = ""

feats_scp = 'feats.scp'
gender_scp = 'gender.scp'
sort_set = []
ff = open(feats_scp, 'r')
gf = open(gender_scp, 'r')
for f_line, g_line in zip(ff.readlines(), gf.readlines()):
    f_line = f_line.strip().split()
    g_line = g_line.strip().split()
    assert f_line[0] == g_line[0]
    sort_set.append((f_line[1], g_line[1]))
batch = 2
train_set = sort_set

def gen():
    global train_set
    for ele in train_set:
        feats = kaldi_io.read_mat(ele[0])
        gender = kaldi_io.read_vec_int(ele[1])
        yield feats, gender
    chunks = [sort_set[i:i+batch] for i in range(0,len(sort_set), batch)]
    random.shuffle(chunks)
    train_set = []
    for i in range(len(chunks)):
        train_set.extend(chunks[i])

def read_test_feats(item):
    file, offset = item.split(':')
    offset = int(offset)
    f = open(file, 'rb')
    f.seek(offset)
    data = np.frombuffer(f.read(), dtype=np.int32)
    f.close()
    return np.reshape(data, [-1,6])

def read_test_gender(item):
    file, offset = item.split(':')
    offset = int(offset)
    f = open(file, 'rb')
    f.seek(offset)
    data = np.frombuffer(f.read(), dtype=np.int32)
    f.close()
    return data


def _slice_example(feats, gender):
    frames = tf.cast(tf.shape(feats)[0] / 3, tf.int32)
    mix = feats[0:frames]
    s1 = feats[frames:frames*2]
    s2 = feats[frames*2:frames*3]
    return mix, s1, s2, frames, gender


dataset = tf.data.Dataset().from_generator(gen, (tf.float32, tf.int32),
                                           (tf.TensorShape([None, None]), tf.TensorShape([None])))
dataset = dataset.map(_slice_example)
dataset = dataset.repeat(2)
padded_shapes = (tf.TensorShape([None, None]),
                 tf.TensorShape([None, None]),
                 tf.TensorShape([None, None]),
                 tf.TensorShape([]),
                 tf.TensorShape([None]))
dataset = dataset.padded_batch(2, padded_shapes=padded_shapes)
iterator = dataset.make_one_shot_iterator()
next_batch = iterator.get_next()

sess = tf.InteractiveSession()
# sess.run(iterator.initializer)
i = 1
while True:
    try:
        mix, s1, s2, frames, gender = sess.run(next_batch)
        print("batch %d:"%i)
        print(mix.shape, s1.shape, s2.shape)
        print(frames)
        print(gender)
        i += 1
    except tf.errors.OutOfRangeError:
        print("end traveling")
        break
