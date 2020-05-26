import sys
from glob import iglob

import numpy as np


timit_dir = sys.argv[1]
output_file = sys.argv[2]

pattern = '{}/**/*.wav'.format(timit_dir)
file_list = iglob(pattern, recursive=True)
file_list = filter(lambda f: not f.endswith('.wav.wav'), file_list)
file_list = map(lambda f: f[len(timit_dir) + 1 :], file_list)
file_list = list(file_list)


def get_speaker(path):
    return path.split('/')[2]


speakers = sorted(list(set(map(get_speaker, file_list))))

speaker_to_id = dict()
for i, s in enumerate(speakers):
    speaker_to_id[s] = i

id_list = map(lambda f: speaker_to_id[get_speaker(f)], file_list)
id_list = list(id_list)

file_path_to_speaker_id = dict()
for f, i in zip(file_list, id_list):
    file_path_to_speaker_id[f] = i

np.save(output_file, file_path_to_speaker_id)
