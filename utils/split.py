import os
import random
import sys

import numpy as np


path_list_file = sys.argv[1]
path_to_label_file = sys.argv[2]
train_path_file = sys.argv[3]
test_path_file = sys.argv[4]

with open(path_list_file, 'r') as f:
    path_list = list(map(str.strip, f.readlines()))
path_to_label = np.load(path_to_label_file, allow_pickle=True).item()


label_to_paths = dict()
for path in path_list:
    label = path_to_label[path]
    if label not in label_to_paths:
        label_to_paths[label] = []
    label_to_paths[label].append(path)

train_samples = []
test_samples = []
for paths in label_to_paths.values():
    random.shuffle(paths)
    n_test = len(paths) // 3
    test_samples.extend(paths[:n_test])
    train_samples.extend(paths[n_test:])


def save_str_list(path, str_list):
    with open(path, 'w') as f:
        for s in str_list:
            print(s, file=f)


save_str_list(train_path_file, train_samples)
save_str_list(test_path_file, test_samples)