import io
import os

import numpy as np
import soundfile as sf
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.utils import to_categorical

import conf


def batchGenerator(conf, fact_amp, out_dim):
    while True:
        sig_batch, lab_batch = create_batches_rnd(conf, fact_amp, out_dim)
        yield sig_batch, lab_batch


def create_batches_rnd(conf, fact_amp, out_dim):
    """
    Initialization of the minibatch
    (batch_size, [0=>x_t, 1=>x_t+N, 1=>random_samp])
    """
    sig_batch = np.zeros([conf.batch_size, conf.wlen])
    lab_batch = []
    snt_id_arr = np.random.randint(conf.snt_tr, size=conf.batch_size)
    rand_amp_arr = np.random.uniform(
        1.0 - fact_amp,
        1 + fact_amp,
        conf.batch_size
    )
    for i in range(conf.batch_size):
        # select a random sentence from the list
        fname = conf.data_folder + conf.wav_lst_tr[snt_id_arr[i]]
        with tf.io.gfile.GFile(fname, 'rb') as f:
            [signal, fs] = sf.read(io.BytesIO(f.read()))
        # accesing to a random chunk
        snt_len = signal.shape[0]
        snt_beg = np.random.randint(snt_len - conf.wlen - 1)
        snt_end = snt_beg + conf.wlen
        sig_batch[i, :] = signal[snt_beg:snt_end] * rand_amp_arr[i]
        y = conf.lab_dict[conf.wav_lst_tr[snt_id_arr[i]]]
        yt = to_categorical(y, num_classes=out_dim)
        lab_batch.append(yt)
    a, b = np.shape(sig_batch)
    sig_batch = sig_batch.reshape((a, b, 1))
    return sig_batch, np.array(lab_batch)


def main():
    K.clear_session()
    # np.random.seed(seed)
    # from tensorflow import set_random_seed
    # set_random_seed(seed)

    input_shape = (conf.wlen, 1)
    out_dim = conf.class_lay[0]
    from model import getModel

    model = getModel(input_shape, out_dim)
    optimizer = RMSprop(lr=conf.lr, rho=0.9, epsilon=1e-8)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    checkpoints_path = os.path.join(conf.output_folder, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoints_path, 'SincNet.hdf5'),
        verbose=1,
        save_best_only=False,
    )
    callbacks = [checkpointer]

    if conf.pt_file != 'none':
        model.load_weights(conf.pt_file)

    train_generator = batchGenerator(conf, 0.2, out_dim)
    model.fit_generator(
        train_generator,
        steps_per_epoch=conf.N_batches,
        epochs=conf.N_epochs,
        verbose=1,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    main()
