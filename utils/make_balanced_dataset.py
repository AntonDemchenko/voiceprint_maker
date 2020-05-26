import os
import random
import sys

import numpy as np


path_list_file = sys.argv[1]
path_to_label_file = sys.argv[2]

with open(path_list_file, 'r') as f:
    path_list = list(map(str.strip, f.readlines()))
path_to_label = np.load(path_to_label_file, allow_pickle=True).item()


label_to_paths = dict()
for path in path_list:
    label = path_to_label[path]
    if label not in label_to_paths:
        label_to_paths[label] = []
    label_to_paths[label].append(path)

lengths = list(map(len, label_to_paths.values()))
samples_per_class = min(lengths)

samples = []
for paths in label_to_paths.values():
    samples.extend(random.sample(paths, k=samples_per_class))

print('\n'.join(samples))