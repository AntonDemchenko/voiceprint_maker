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


def read_wav(path):
    with tf.io.gfile.GFile(path, 'rb') as f:
        signal, _ = sf.read(io.BytesIO(f.read()))
        return signal


def get_label(path, cfg):
    label = cfg.lab_dict[path]
    return to_categorical(label, num_classes=cfg.out_dim)


def get_sample(path, cfg):
    full_path = cfg.data_folder + path
    signal = read_wav(full_path)
    chunk_begin = np.random.randint(signal.shape[0] - cfg.wlen)
    signal = signal[chunk_begin : chunk_begin + cfg.wlen]
    amp = np.random.uniform(1.0 - cfg.fact_amp, 1.0 + cfg.fact_amp)
    signal = signal * amp
    label = get_label(path, cfg)
    return signal.reshape((signal.shape[0], 1)), label


def sample_reader(cfg):
    for path in cfg.train_list:
        yield get_sample(path, cfg)


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
        lambda: sample_reader(cfg), 
        (tf.float64, tf.int64), 
        (tf.TensorShape([cfg.wlen, 1]), tf.TensorShape([cfg.out_dim])),
    ).shuffle(256).repeat().batch(cfg.batch_size).prefetch(2)
    model.fit(
        train_dataset,
        steps_per_epoch=cfg.N_batches,
        epochs=cfg.N_epochs,
        verbose=1,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    main()
