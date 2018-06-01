import os, sys
lib_path = os.path.abspath('..')
sys.path.append(lib_path)

import numpy as np
from utils.kaldi_io import read_mat

decay = 0.9999
data_type = ['tr', 'cv', 'tt']
feats_dim = 129
dir_pattern = "/mnt/hd8t/fsl/SpeechSeparation/mix/data/2speakers/wav8k/min/{}/record"

def moving_average(mean_var, variance_var, value, decay):
    variance_var = decay * (variance_var + (1 - decay) * (value - mean_var) ** 2)
    mean_var = decay * mean_var + (1 - decay) * value
    return mean_var, variance_var

for i_type in data_type:
    dir_path = dir_pattern.format(i_type)
    cmvn_path = os.path.join(dir_path, 'global.cmvn.log')
    feats_path = os.path.join(dir_path, 'feats.scp')
    mean_var = np.zeros(feats_dim)
    variance_var = np.zeros(feats_dim)
    with open(feats_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            feats = read_mat(line[1])
            frames = int(len(feats) / 3)
            # use log(1+x)
            log_feats_input = np.log(feats[0:frames, 0:feats_dim] + 1)
            for item in log_feats_input:
                mean_var, variance_var = moving_average(mean_var, variance_var,
                                                        item, decay)
    cmvn = np.stack([mean_var, variance_var], axis=1)
    np.savetxt(cmvn_path, cmvn)