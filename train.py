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


def batchGenerator(cfg, fact_amp):
    while True:
        sig_batch, lab_batch = create_batches_rnd(cfg, fact_amp)
        yield sig_batch, lab_batch


def create_batches_rnd(cfg, fact_amp):
    """
    Initialization of the minibatch
    (batch_size, [0=>x_t, 1=>x_t+N, 1=>random_samp])
    """
    sig_batch = np.zeros([cfg.batch_size, cfg.wlen])
    lab_batch = []
    snt_id_arr = np.random.randint(cfg.snt_tr, size=cfg.batch_size)
    rand_amp_arr = np.random.uniform(
        1.0 - fact_amp,
        1 + fact_amp,
        cfg.batch_size
    )
    for i in range(cfg.batch_size):
        # select a random sentence from the list
        fname = cfg.data_folder + cfg.train_list[snt_id_arr[i]]
        with tf.io.gfile.GFile(fname, 'rb') as f:
            [signal, fs] = sf.read(io.BytesIO(f.read()))
        # accesing to a random chunk
        snt_len = signal.shape[0]
        snt_beg = np.random.randint(snt_len - cfg.wlen - 1)
        snt_end = snt_beg + cfg.wlen
        sig_batch[i, :] = signal[snt_beg:snt_end] * rand_amp_arr[i]
        y = cfg.lab_dict[cfg.train_list[snt_id_arr[i]]]
        yt = to_categorical(y, num_classes=cfg.out_dim)
        lab_batch.append(yt)
    a, b = np.shape(sig_batch)
    sig_batch = sig_batch.reshape((a, b, 1))
    return sig_batch, np.array(lab_batch)


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

    train_generator = batchGenerator(cfg, 0.2)
    model.fit_generator(
        train_generator,
        steps_per_epoch=cfg.N_batches,
        epochs=cfg.N_epochs,
        verbose=1,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    main()
