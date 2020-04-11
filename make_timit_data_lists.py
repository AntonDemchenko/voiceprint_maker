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
    validation_list_file = os.path.join(output_dir, '{}_validation.scp'.format(dataset_name))
    test_list_file = os.path.join(output_dir, '{}_test.scp'.format(dataset_name))

    pattern = '{}/**/*.wav'.format(dataset_dir)
    all_list = iglob(pattern, recursive=True)
    all_list = filter(lambda f: not f.endswith('.wav.wav'), all_list)
    all_list = map(lambda f: f[len(dataset_dir) + 1 :], all_list)
    all_list = list(all_list)

    def get_speaker(path):
        return path.split('/')[2]

    speaker_to_file_list = dict()
    for path in all_list:
        speaker = get_speaker(path)
        if speaker not in speaker_to_file_list:
            speaker_to_file_list[speaker] = list()
        speaker_to_file_list[speaker].append(path)

    for speaker in speaker_to_file_list:
        np.random.shuffle(speaker_to_file_list[speaker])

    test_list = []
    validation_list = []
    train_list = []
    test_samples_per_speaker = 3
    val_samples_per_speaker = 1
    for speaker in speaker_to_file_list:
        l = speaker_to_file_list[speaker]
        test_list.extend(l[:test_samples_per_speaker])
        l = l[test_samples_per_speaker:]
        validation_list.extend(l[:val_samples_per_speaker])
        l = l[val_samples_per_speaker:]
        train_list.extend(l)

    save_str_list(all_list_file, all_list)
    save_str_list(test_list_file, test_list)
    save_str_list(train_list_file, train_list)
    save_str_list(validation_list_file, validation_list)


def save_str_list(path, str_list):
    with open(path, 'w') as f:
        for s in str_list:
            print(s, file=f)


if __name__ == '__main__':
    main()
