import sys, os
libpath = os.path.abspath('..')
sys.path.append(libpath)

from utils.signalprocess import audioread, stft
from utils.kaldi_io import write_mat, write_vec_int
import numpy as np


list_pattern = "/mnt/hd8t/fsl/SpeechSeparation/mix/mix3spk/mix_3_spk_min_{}_mix.time.sort"
wav_dir = "/mnt/hd8t/fsl/SpeechSeparation/mix/data/3speakers/wav8k/min"
data_type = ['tr', 'cv', 'tt']
pre_pattern = "mix_3_spk_min_"
fs8k = 8000
size = 256
shift = 64
feat_dims = size // 2 + 1
decay = 0.9999

info_list = "spkrinfo.txt"
gender_dict = {}
for line in open(info_list, 'r').readlines():
    line = line.strip().split()
    gender_dict[line[0]] = int(line[1] == 'M')  # gender=1 if Male


def moving_average(mean_var, variance_var, value, decay):
    variance_var = decay * (variance_var + (1 - decay) * (value - mean_var) ** 2)
    mean_var = decay * mean_var + (1 - decay) * value
    return mean_var, variance_var


for i_type in data_type:
    print('start extracting features from directory ' + i_type)
    list_file = list_pattern.format(i_type)
    record_dir = os.path.join(wav_dir, i_type, 'record')
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    write_ff = open(os.path.join(record_dir, 'feats.ark'), 'wb')
    write_gf = open(os.path.join(record_dir, 'gender.ark'), 'wb')
    write_lenf = open(os.path.join(record_dir, 'feats.len'), 'w')
    mean_var_file = os.path.join(record_dir, 'global.cmvn')
    feat_mean = np.zeros(feat_dims, dtype=np.float32)
    feat_variance = np.ones(feat_dims, dtype=np.float32)
    with open(list_file, 'r') as list_f:
        lines = list_f.readlines()
        for i_line in range(len(lines)):
            line = lines[i_line]
            line = line.strip().split()
            key = line[0]
            mix_file = os.path.join(wav_dir, i_type, 'mix', key + '.wav')
            s1_file = os.path.join(wav_dir, i_type, 's1', key + '.wav')
            s2_file = os.path.join(wav_dir, i_type, 's2', key + '.wav')
            s3_file = os.path.join(wav_dir, i_type, 's3', key + '.wav')

            mix_wav = audioread(mix_file, samp_rate=fs8k)
            s1_wav = audioread(s1_file, samp_rate=fs8k)
            s2_wav = audioread(s2_file, samp_rate=fs8k)
            s3_wav = audioread(s3_file, samp_rate=fs8k)

            s1_gender = gender_dict[key.split('_')[0][0:3]]
            s2_gender = gender_dict[key.split('_')[2][0:3]]
            s3_gender = gender_dict[key.split('_')[4][0:3]]
            gender = np.array([s1_gender, s2_gender, s3_gender]).astype(np.int32)
            write_vec_int(write_gf, gender, key=key)

            mix_stft = stft(mix_wav, size=size, shift=shift).astype(np.complex64)
            mix_abs = np.abs(mix_stft)
            mix_angle = np.angle(mix_stft)
            s1_stft = stft(s1_wav, size=size, shift=shift).astype(np.complex64)
            s1_abs = np.abs(s1_stft)
            s1_angle = np.angle(s1_stft)
            s2_stft = stft(s2_wav, size=size, shift=shift).astype(np.complex64)
            s2_abs = np.abs(s2_stft)
            s2_angle = np.angle(s2_stft)
            s3_stft = stft(s3_wav, size=size, shift=shift).astype(np.complex64)
            s3_abs = np.abs(s3_stft)
            s3_angle = np.angle(s3_stft)

            num_frames = mix_stft.shape[0]
            write_lenf.write(key + ' ' + str(num_frames) + '\n')
            for i in range(num_frames):
                feat_mean, feat_variance = moving_average(feat_mean, feat_variance,
                                                          mix_abs[i], decay)

            mix_data = np.concatenate((mix_abs, mix_angle), axis=1)
            s1_data = np.concatenate((s1_abs, s1_angle), axis=1)
            s2_data = np.concatenate((s2_abs, s2_angle), axis=1)
            s3_data = np.concatenate((s3_abs, s3_angle), axis=1)
            feats = np.concatenate((mix_data, s1_data, s2_data, s3_data), axis=0).astype(np.float32)
            write_mat(write_ff, feats, key=key)
            if (i_line + 1) % 1000 == 0:
                print('processed %d sentence' % (i_line + 1))
    write_ff.close()
    write_gf.close()
    write_lenf.close()
    mean_variance = np.stack((feat_mean, feat_variance), axis=1)
    np.savetxt(mean_var_file, mean_variance)
    print('finished task for directory ' + i_type)
