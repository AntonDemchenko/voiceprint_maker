import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def make_train_dataset(self, path_list):
        np.random.shuffle(path_list)
        path_dataset = tf.data.Dataset.from_tensor_slices(path_list)
        signal_dataset = path_dataset\
            .map(self.read_signal, tf.data.experimental.AUTOTUNE)\
            .cache()\
            .map(self.random_crop, tf.data.experimental.AUTOTUNE)
        label_dataset = self.make_label_dataset(path_list)
        input_dataset = tf.data.Dataset\
            .zip((signal_dataset, label_dataset))
        signal_label_dataset = tf.data.Dataset\
            .zip((input_dataset, label_dataset))\
            .shuffle(len(path_list))\
            .repeat()\
            .batch(self.cfg.batch_size)\
            .prefetch(tf.data.experimental.AUTOTUNE)
        return signal_label_dataset

    def make_val_dataset(self, path_list):
        path_dataset = tf.data.Dataset.from_tensor_slices(path_list)
        signal_dataset = path_dataset\
            .map(self.read_signal, tf.data.experimental.AUTOTUNE)\
            .cache()\
            .repeat(self.cfg.n_val_windows_per_sample)\
            .map(self.random_crop, tf.data.experimental.AUTOTUNE)
        label_dataset = self.make_label_dataset(path_list)\
            .repeat(self.cfg.n_val_windows_per_sample)
        input_dataset = tf.data.Dataset\
            .zip((signal_dataset, label_dataset))
        signal_label_dataset = tf.data.Dataset\
            .zip((input_dataset, label_dataset))\
            .batch(self.cfg.batch_size_test)\
            .cache()\
            .prefetch(tf.data.experimental.AUTOTUNE)
        return signal_label_dataset

    def make_test_iterable(self, path_list):
        def get_test_samples(path_list):
            for path in path_list:
                signal = self.read_signal(path)
                for chunk in self.make_test_chunks(signal):
                    yield path, chunk
        path_batch = []
        signal_batch = []
        for path, signal in get_test_samples(path_list):
            path_batch.append(path)
            signal_batch.append(signal)
            if len(path_batch) == self.cfg.batch_size_test:
                yield np.array(path_batch), tf.convert_to_tensor(signal_batch)
                path_batch = []
                signal_batch = []
        if path_batch:
            yield np.array(path_batch), tf.convert_to_tensor(signal_batch)

    def make_test_dataset(self, path_list):
        dataset = tf.data.Dataset\
            .from_generator(
                lambda: self.get_test_samples(path_list),
                ((tf.float32, tf.int32), tf.int32),
                (([self.cfg.window_len, 1], [self.cfg.n_classes]), [self.cfg.n_classes])
            )\
            .batch(self.cfg.batch_size_test)\
            .cache()\
            .prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def get_test_samples(self, path_list):
        for path in path_list:
            signal = self.read_signal(path)
            label = self.transform_path_to_label(path)
            for chunk in self.make_test_chunks(signal):
                yield ((chunk, label), label)

    def make_test_chunks(self, signal):
        for chunk_begin in range(0, signal.shape[0] - self.cfg.window_len + 1, self.cfg.window_shift):
            chunk = signal[chunk_begin : chunk_begin + self.cfg.window_len]
            yield chunk

    def read_signal(self, path):
        wav = self.read_wav(path)
        return self.decode_wav(wav)

    def read_wav(self, path):
        full_path = tf.strings.join([self.cfg.dataset_folder, path], separator='/')
        return tf.io.read_file(full_path)

    def decode_wav(self, wav):
        signal = tf.audio.decode_wav(wav, desired_channels=1)[0]
        signal = tf.slice(signal, [0, 0], [-1, 1])
        return signal

    def random_crop(self, signal):
        return tf.image.random_crop(signal, [self.cfg.window_len, 1])

    def random_change_amplitude(self, signal_batch, label_batch):
        amp = tf.random.uniform(
            [self.cfg.window_len, 1],
            minval=1.0 - self.cfg.fact_amp,
            maxval=1.0 + self.cfg.fact_amp,
            dtype=tf.float32
        )
        signal_batch[0] *= amp
        return (signal_batch, label_batch)

    def label_to_categorical(self, label):
        return to_categorical(label, num_classes=self.cfg.n_classes)

    def make_label_dataset(self, path_list):
        return tf.data.Dataset.from_tensor_slices(
            [
                self.transform_path_to_label(path)
                for path in path_list
            ]
        )

    def transform_path_to_label(self, path):
        label = self.cfg.path_to_label[path]
        return to_categorical(label, num_classes=self.cfg.n_classes)