import os
import sys

import numpy as np


path_prefix = sys.argv[1]
path_list_file = sys.argv[2]
path_to_label_file = sys.argv[3]

with open(path_list_file, 'r') as f:
    path_list = list(map(str.strip, f.readlines()))
path_to_label = np.load(path_to_label_file, allow_pickle=True).item()


label_to_cnt = dict()
for path in path_list:
    label = path_to_label[path]
    if label not in label_to_cnt:
        label_to_cnt[label] = 0
    label_to_cnt[label] += 1

counts = list(label_to_cnt.values())
counts.sort()

print('Set size', len(path_list))
print('Classes', len(label_to_cnt))
print('Min', min(counts))
print('Median', counts[len(counts) // 2])
print('Max', max(counts))
print('Mean', sum(counts) / len(counts))
print('Sum', sum(counts))
