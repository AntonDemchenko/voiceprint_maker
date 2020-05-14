import time
import sys

import tensorflow as tf
import numpy as np

from config import read_config
from data_loader import DataLoader
from sincnet import create_print_maker


def split_to_batches(array, batch_size):
    batches = []
    for i in range(0, len(array), batch_size):
        batch = array[i : i + batch_size]
        batch = tf.convert_to_tensor(batch)
        batches.append(batch)
    return batches


def make_voiceprint(model, data_loader, file_name):
    signal = data_loader.read_signal(file_name).numpy()
    windows = list(data_loader.split_to_windows(signal))
    batches = split_to_batches(windows, data_loader.cfg.batch_size_test)
    window_voiceprints = []
    for batch in batches:
        voiceprint_batch = model.predict(batch)
        window_voiceprints.extend(voiceprint_batch)
    window_voiceprints = np.array(window_voiceprints)
    voiceprint = np.mean(window_voiceprints, axis=0)
    voiceprint = tf.math.l2_normalize(voiceprint).numpy()
    return voiceprint


def main():
    cfg = read_config()
    data_loader = DataLoader(cfg)
    model = create_print_maker(cfg)
    for layer in model.layers:
        layer.trainable = False

    for file_name in map(str.strip, sys.stdin):
        start = time.time()
        voiceprint = make_voiceprint(model, data_loader, file_name)
        print(time.time() - start, 'sec')
        # output = ' '.join(map(str, voiceprint))
        # print(output)


if __name__ == '__main__':
    main()