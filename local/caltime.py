import wave
import os
import tensorflow as tf

dir_root = "/mnt/hd8t/fsl/SpeechSeparation/mix"
data_root = "data/2speakers/wav8k/min"
data_type = ['tr', 'cv', 'tt']
name_pre = "mix2spk/mix_2_spk_min_"

for i_type in data_type:
    name_path = name_pre + i_type + '_mix'
    data_path = os.path.join(data_root, i_type, 'mix')
    name_list = os.path.join(dir_root, name_path)
    write_path = name_list + '.time.sort'
    write_file = open(write_path, 'w')
    tuple_list = []
    with open(name_list, 'r') as f:
        for line in f.readlines():
            key = line.strip()
            key_name = key + '.wav'
            file_path = os.path.join(dir_root, data_path, key_name)
            wave_read = wave.open(file_path)
            samprate = wave_read.getframerate()
            nframes = wave_read.getnframes()
            duration = nframes / samprate
            tuple_list.append((key, duration))
            wave_read.close()
    tuple_list = sorted(tuple_list, key=lambda x: x[1])
    for ele in tuple_list:
        write_file.write(ele[0] + ' ' + str(round(ele[1], 2)) + '\n')
    write_file.close()
