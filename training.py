import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard


def make_optimizer(cfg):
    from tensorflow.keras import optimizers

    if cfg.optimizer == 'rmsprop':
        return optimizers.RMSprop(learning_rate=cfg.lr, rho=0.95, epsilon=1e-8)
    if cfg.optimizer == 'adam':
        return optimizers.Adam(learning_rate=cfg.lr)
    if cfg.optimizer == 'adagrad':
        return optimizers.Adagrad(learning_rate=cfg.lr)
    if cfg.optimizer == 'sgd':
        return optimizers.SGD(learning_rate=cfg.lr)


def initialize_session(cfg):
    K.clear_session()
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)


def make_callbacks(cfg):
    checkpoints_path = os.path.join(cfg.output_folder, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoints_path, cfg.checkpoint_name),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        period=cfg.checkpoint_freq
    )

    last_checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoints_path, 'last_checkpoint.hdf5'),
        verbose=0,
        save_weights_only=True,
        period=1
    )

    csv_path = os.path.join(cfg.output_folder, 'log.csv')
    csv_logger = CSVLogger(csv_path, append=(cfg.initial_epoch > 0))

    logs_path = os.path.join(cfg.output_folder, 'logs')
    tensorboard_logger = TensorBoard(logs_path, write_graph=False, profile_batch=0)

    return [checkpointer, last_checkpointer, tensorboard_logger, csv_logger]


def train(cfg, model, data_loader):
    callbacks = make_callbacks(cfg)

    train_dataset = data_loader.make_train_dataset(cfg.train_list)
    validation_dataset = data_loader.make_validation_dataset(cfg.validation_list)
    model.fit(
        train_dataset,
        steps_per_epoch=cfg.N_batches,
        initial_epoch=cfg.initial_epoch,
        epochs=cfg.N_epochs,
        verbose=2,
        callbacks=callbacks,
        validation_data=validation_dataset,
        validation_freq=cfg.N_eval_epoch
    )
