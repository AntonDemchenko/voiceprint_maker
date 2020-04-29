import os
import sys
from glob import iglob

import numpy as np


def main():
    np.random.seed(1234)

    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2]

    dataset_name = os.path.basename(dataset_dir)

    all_list_file = os.path.join(output_dir, '{}_all.scp'.format(dataset_name))
    train_list_file = os.path.join(output_dir, '{}_train.scp'.format(dataset_name))
    val_list_file = os.path.join(output_dir, '{}_validation.scp'.format(dataset_name))
    test_list_file = os.path.join(output_dir, '{}_test.scp'.format(dataset_name))

    pattern = '{}/**/*.wav'.format(dataset_dir)
    all_list = iglob(pattern, recursive=True)
    all_list = filter(lambda f: not f.endswith('.wav.wav'), all_list)
    all_list = map(lambda f: f[len(dataset_dir) + 1 :], all_list)
    all_list = list(all_list)

    test_fraction = 0.25
    validation_fraction = 0.15
    test_list, train_val_list = split(all_list, test_fraction)
    val_list, train_list = split(train_val_list, validation_fraction)

    save_str_list(all_list_file, all_list)
    save_str_list(test_list_file, test_list)
    save_str_list(train_list_file, train_list)
    save_str_list(val_list_file, val_list)


def split(source_list, left_fraction):
    left_list = []
    right_list = []
    for path in source_list:
        if np.random.random() < left_fraction:
            left_list.append(path)
        else:
            right_list.append(path)
    return left_list, right_list


def save_str_list(path, str_list):
    with open(path, 'w') as f:
        for s in str_list:
            print(s, file=f)


if __name__ == '__main__':
    main()
