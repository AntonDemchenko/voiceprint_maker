import io

import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


def read_wav(path):
    with tf.io.gfile.GFile(path, 'rb') as f:
        signal, _ = sf.read(io.BytesIO(f.read()))
        return signal


def get_label(cfg, path):
    label = cfg.lab_dict[path]
    # return to_categorical(label, num_classes=cfg.n_classes)
    return label


def get_training_sample(cfg, path):
    full_path = cfg.data_folder + path
    signal = read_wav(full_path)
    chunk_begin = np.random.randint(signal.shape[0] - cfg.wlen + 1)
    chunk = signal[chunk_begin : chunk_begin + cfg.wlen]
    amp = np.random.uniform(1.0 - cfg.fact_amp, 1.0 + cfg.fact_amp)
    chunk = chunk * amp
    label = get_label(cfg, path)
    yield chunk.reshape((chunk.shape[0], 1)), label


def get_testing_samples(cfg, path):
    full_path = cfg.data_folder + path
    signal = read_wav(full_path)
    label = get_label(cfg, path)
    for chunk_begin in range(0, signal.shape[0] - cfg.wlen + 1, cfg.wshift):
        chunk = signal[chunk_begin : chunk_begin + cfg.wlen]
        yield chunk.reshape((chunk.shape[0], 1)), label


def sample_reader(cfg, path_list, get_samples):
    for path in path_list:
        yield from get_samples(cfg, path)


def make_dataset(cfg, path_list, for_train=True):
    get_samples = get_training_sample if for_train else get_testing_samples

    def get_generator():
        return sample_reader(cfg, path_list, get_samples)

    dataset = tf.data.Dataset.from_generator(
        get_generator,
        (tf.float32, tf.int32),
        (tf.TensorShape([cfg.wlen, 1]), tf.TensorShape([])),
    )
    if for_train:
        dataset = dataset.shuffle(1024).repeat()
    dataset = dataset.batch(cfg.batch_size)
    return dataset
