# config file for training or other process
# check the config file before your process
# and import this file to get parameters

# data
train_dir = "/mnt/hd8t/fsl/SpeechSeparation/mix/data/2speakers/wav8k/min/tr/cut_15k_record"
valid_dir = "/mnt/hd8t/fsl/SpeechSeparation/mix/data/2speakers/wav8k/min/cv/cut_4k_record"

# job
seed = 123
gpu = '0'
job_dir = "job/Anchor_6anc_4x300blstm_2spkr_20embed_dropout_psloss"
job_type = "eval"
resume = False

# feat config
frame_size = 256
shift = 64
feats_dim = frame_size // 2 + 1
window_type = 'hann'
global_cmvn_norm = True
global_cmvn_file = "/mnt/hd8t/fsl/SpeechSeparation/mix/data/2speakers/wav8k/" \
                   "min/tr/cut_15k_record/global.cmvn.log"

# model config
input_style = 0
MAX_SOURCE_NUM = 2

# encoder
embedding_dim = 20
encoder_rnn_type = "lstm"
bidirectional = True
encoder_layers = 4
encoder_dim = 300
fw_dropout_keep = 0.5
recur_dropout_keep = 0.8

# estimator
mask_type = "irm"  # supported type [ibm, irm, wfm]
estimator_type = "anchor"  # supported type [avg, avg_thresh, avg_weighted, anchor]
threshold = 0.1
num_anchors = 6

# separator
separator_activation = 'softmax'  # supported type [softmax, sigmoid]


# training param
batch_size = 25
MAX_EPOCHS = 100
start_shuffle_epoch = 1
train_log_freq = 1
valid_freq = 500
save_freq = 500
learning_rate = 0.0005
min_learning_rate = 1e-6
clip_grad = 200
early_stop_count = 6
decay_lr_count = 3
decay_lr = 0.5
weight_decay = 0.00001
weight_decay_exclude = 'Bias:|b:|biases:|BatchNorm|anchors'
loss_phase_sense = False

# eval param
eval_dir = "/mnt/hd8t/fsl/SpeechSeparation/mix/data/2speakers/wav8k/min/tt/record"
eval_name = "eval_wsj_aDAN"
eval_estimator_type = 'anchor'  # supported ['anchor', 'kmeans']
test_wav = "/mnt/hd8t/fsl/SpeechSeparation/mix/data/2speakers/wav8k/" \
           "min/tt/mix/050a0501_1.7783_442o030z_-1.7783.wav"

# load option
load_option = 1
load_path = ""
