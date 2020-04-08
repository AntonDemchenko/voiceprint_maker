import io
import os

import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical

from config import read_config
from sincnet import SincNetModel


def batch_generator(cfg):
    while True:
        signal_batch, label_batch = get_random_batch(cfg)
        yield signal_batch, label_batch


def get_random_chunk(signal, cfg):
    amp = np.random.uniform(1.0 - cfg.fact_amp, 1.0 + cfg.fact_amp)
    signal_len = signal.shape[0]
    begin = np.random.randint(signal_len - cfg.wlen)
    end = begin + cfg.wlen
    return signal[begin:end] * amp


def get_sample(path, cfg):
    full_path = cfg.data_folder + path
    with tf.io.gfile.GFile(full_path, 'rb') as f:
        [signal, _] = sf.read(io.BytesIO(f.read()))
    label = to_categorical(cfg.lab_dict[path], num_classes=cfg.out_dim)
    return signal, label


def get_sample_batch(path_batch, cfg):
    signal_batch = []
    label_batch = []
    for path in path_batch:
        signal, label = get_sample(path, cfg)
        signal = get_random_chunk(signal, cfg)
        signal_batch.append(signal)
        label_batch.append(label)
    signal_batch = np.array(signal_batch).reshape((cfg.batch_size, cfg.wlen, 1))
    label_batch = np.array(label_batch)
    return signal_batch, label_batch


def get_random_batch(cfg):
    chosen_indexes = np.random.randint(len(cfg.train_list), size=cfg.batch_size)
    path_batch = np.array(cfg.train_list)[chosen_indexes]
    return get_sample_batch(path_batch, cfg)


def main():
    cfg = read_config()

    K.clear_session()

    model = SincNetModel(cfg)

    optimizer = RMSprop(lr=cfg.lr, rho=0.9, epsilon=1e-8)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    checkpoints_path = os.path.join(cfg.output_folder, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoints_path, 'SincNet.hdf5'),
        verbose=1,
        save_best_only=False,
    )
    callbacks = [checkpointer]

    if cfg.pt_file != 'none':
        model.load_weights(cfg.pt_file)

    train_dataset = tf.data.Dataset.from_generator(
        batch_generator, 
        (tf.float64, tf.int64), 
        (tf.TensorShape([cfg.batch_size, cfg.wlen, 1]), tf.TensorShape([cfg.batch_size, cfg.out_dim])),
        args=(cfg,)
    )
    model.fit(
        train_dataset,
        steps_per_epoch=cfg.N_batches,
        epochs=cfg.N_epochs,
        verbose=1,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    main()
