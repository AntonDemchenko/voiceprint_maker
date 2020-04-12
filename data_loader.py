import io

import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


def read_wav(path):
    with tf.io.gfile.GFile(path, 'rb') as f:
        signal, _ = sf.read(io.BytesIO(f.read()))
        return signal


def sample_reader(path_list, get_samples):
    for path in path_list:
        yield from get_samples(path)


class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def make_train_dataset(self, path_list):
        path_dataset = tf.data.Dataset.from_tensor_slices(path_list)
        wav_dataset = path_dataset\
            .map(self.read_wav)\
            .cache()
        signal_dataset = wav_dataset\
            .map(self.decode_wav)\
            .map(self.random_crop)\
            .map(self.random_change_amplitude)\
            .map(self.reshape_signal)
        # dataset = dataset.shuffle(len(path_list))

        label_dataset = tf.data.Dataset.from_tensor_slices(
            [
                self.label_to_categorical(self.cfg.lab_dict[path])
                for path in path_list
            ]
        )

        signal_label_dataset = tf.data.Dataset\
            .zip((signal_dataset, label_dataset))\
            .repeat()\
            .batch(self.cfg.batch_size)

        return signal_label_dataset

        # dataset = tf.data.Dataset.from_generator(
        #     lambda: sample_reader(path_list, self.get_train_sample),
        #     (tf.float32, tf.int32),
        #     self.get_output_shape(),
        # )
        # dataset = dataset.shuffle(1024).repeat()
        # dataset = dataset.batch(self.cfg.batch_size)
        # return dataset

    def read_wav(self, path):
        full_path = tf.strings.join([self.cfg.data_folder, path], separator='/')
        return tf.io.read_file(full_path)

    def decode_wav(self, wav):
        signal = tf.audio.decode_wav(wav, desired_channels=1)[0]
        signal = tf.reshape(signal, [1, -1])
        return signal[0]

    def random_crop(self, signal):
        return tf.image.random_crop(signal, [self.cfg.wlen])

    def random_change_amplitude(self, signal):
        amp = tf.random.uniform(
            [], 
            minval=1.0 - self.cfg.fact_amp,
            maxval=1.0 + self.cfg.fact_amp, 
            dtype=tf.float32
        )
        return signal * amp

    def reshape_signal(self, signal):
        return tf.reshape(signal, [self.cfg.wlen, 1])

    def label_to_categorical(self, label):
        return to_categorical(label, num_classes=self.cfg.n_classes)

    def make_test_dataset(self, path_list):
        dataset = tf.data.Dataset.from_generator(
            lambda: sample_reader(path_list, self.get_test_samples),
            (tf.float32, tf.int32),
            self.get_output_shape(),
        )
        dataset = dataset.batch(self.cfg.batch_size_test)
        return dataset

    def get_label(self, path):
        raise NotImplementedError

    def get_output_shape(self):
        raise NotImplementedError

    def get_train_sample(self, path):
        full_path = self.cfg.data_folder + path
        signal = read_wav(full_path)
        chunk_begin = np.random.randint(signal.shape[0] - self.cfg.wlen + 1)
        chunk = signal[chunk_begin : chunk_begin + self.cfg.wlen]
        amp = np.random.uniform(1.0 - self.cfg.fact_amp, 1.0 + self.cfg.fact_amp)
        chunk = chunk * amp
        label = self.get_label(path)
        yield chunk.reshape((chunk.shape[0], 1)), label

    def get_test_samples(self, path):
        full_path = self.cfg.data_folder + path
        signal = read_wav(full_path)
        label = self.get_label(path)
        for chunk_begin in range(0, signal.shape[0] - self.cfg.wlen + 1, self.cfg.wshift):
            chunk = signal[chunk_begin : chunk_begin + self.cfg.wlen]
            yield chunk.reshape((chunk.shape[0], 1)), label


class ClassifierDataLoader(DataLoader):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_output_shape(self):
        return (
            tf.TensorShape([self.cfg.wlen, 1]),
            tf.TensorShape([self.cfg.n_classes])
        )

    def get_label(self, path):
        label = self.cfg.lab_dict[path]
        return to_categorical(label, num_classes=self.cfg.n_classes)


class PrintMakerDataLoader(DataLoader):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_output_shape(self):
        return (
            tf.TensorShape([self.cfg.wlen, 1]),
            tf.TensorShape([])
        )

    def get_label(self, path):
        label = self.cfg.lab_dict[path]
        return label

    def make_test_dataset(self, path_list):
        samples = list(sample_reader(path_list, self.get_test_samples))
        np.random.shuffle(samples)
        signal_list = np.array([s[0] for s in samples]).astype(np.float32)
        label_list = np.array([s[1] for s in samples]).astype(np.int32)
        signal_tensor = tf.convert_to_tensor(signal_list)
        label_tensor = tf.convert_to_tensor(label_list)
        dataset = tf.data.Dataset.from_tensor_slices(
            (signal_tensor, label_tensor)
        )
        dataset = dataset.batch(self.cfg.batch_size_test)
        return dataset