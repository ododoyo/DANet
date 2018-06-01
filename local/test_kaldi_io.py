import os, sys
lib_path = os.path.abspath('..')
sys.path.append(lib_path)

from utils import kaldi_io
from utils.signalprocess import *
import numpy as np

wav_dir = "/mnt/hd8t/fsl/SpeechSeparation/mix/data/2speakers/wav8k/min/tr"
test_list = "test_wav.list"

gender_info = "spkrinfo.txt"
gender_dict = {}
for line in open(gender_info, 'r').readlines():
    line = line.strip().split()
    gender_dict[line[0]] = int(line[1] == 'M')

feats_ark = 'test_feats.ark'
gender_ark = 'test_gender.ark'
feats_f = open(feats_ark, 'wb')
gender_f = open(gender_ark, 'wb')
with open(test_list, 'r') as f:
    for line in f.readlines():
        key = line.strip()
        file_name = key + '.wav'
        mix_file = os.path.join(wav_dir, 'mix', file_name)
        s1_file = os.path.join(wav_dir, 's1', file_name)
        s2_file = os.path.join(wav_dir, 's2', file_name)

        mix_wav = audioread(mix_file, samp_rate=8000)
        s1_wav = audioread(s1_file, samp_rate=8000)
        s2_wav = audioread(s2_file, samp_rate=8000)

        s1_gender = gender_dict[key.split('_')[0][0:3]]
        s2_gender = gender_dict[key.split('_')[2][0:3]]
        gender = np.array([s1_gender, s2_gender]).astype(np.int32)
        kaldi_io.write_vec_int(gender_f, gender, key=key)

        mix_stft = stft(mix_wav, size=256, shift=64)
        mix_abs = np.abs(mix_stft)
        mix_angle = np.angle(mix_stft)
        s1_stft = stft(s1_wav, size=256, shift=64)
        s1_abs = np.abs(s1_stft)
        s1_angle = np.angle(s1_stft)
        s2_stft = stft(s2_wav, size=256, shift=64)
        s2_abs = np.abs(s2_stft)
        s2_angle = np.angle(s2_stft)

        mix_data = np.concatenate((mix_abs, mix_angle), axis=1)
        s1_data = np.concatenate((s1_abs, s1_angle), axis=1)
        s2_data = np.concatenate((s2_abs, s2_angle), axis=1)
        feats = np.concatenate((mix_data, s1_data, s2_data), axis=0).astype(np.float32)
        kaldi_io.write_mat(feats_f, feats, key=key)
feats_f.close()
gender_f.close()