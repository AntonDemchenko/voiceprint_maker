import os
import sys
from glob import iglob

import numpy as np


dataset_dir = sys.argv[1]
output_dir = sys.argv[2]

dataset_name = os.path.basename(dataset_dir)

all_list_file = os.path.join(output_dir, '{}_all.scp'.format(dataset_name))
train_list_file = os.path.join(output_dir, '{}_train.scp'.format(dataset_name))
validation_list_file = os.path.join(output_dir, '{}_validation.scp'.format(dataset_name))
test_list_file = os.path.join(output_dir, '{}_test.scp'.format(dataset_name))

pattern = '{}/**/*.wav'.format(dataset_dir)
all_list = iglob(pattern, recursive=True)
all_list = filter(lambda f: not f.endswith('.wav.wav'), all_list)
all_list = map(lambda f: f[len(dataset_dir) + 1 :], all_list)
all_list = list(all_list)


test_list = filter(lambda p: p.startswith('test'), all_list)
test_list = list(test_list)

train_val_list = filter(lambda p: p.startswith('train'), all_list)

fraction = 0.15
np.random.seed(1234)

train_list = []
validation_list = []

for path in train_val_list:
    if np.random.random() < fraction:
        validation_list.append(path)
    else:
        train_list.append(path)


def save_str_list(path, str_list):
    with open(path, 'w') as f:
        for s in str_list:
            print(s, file=f)


save_str_list(all_list_file, all_list)
save_str_list(test_list_file, test_list)
save_str_list(train_list_file, train_list)
save_str_list(validation_list_file, validation_list)