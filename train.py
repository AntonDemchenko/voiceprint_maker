import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

from config import read_config
from data_loader import DataLoader
from sincnet import create_classifier


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


def make_model(cfg):
    model = create_classifier(cfg)
    if cfg.checkpoint_file != 'none':
        model.load_weights(cfg.checkpoint_file)
    optimizer = make_optimizer(cfg)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


def initialize_session(cfg):
    K.clear_session()
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)


def make_best_checkpointer(cfg):
    return ModelCheckpoint(
        filepath=cfg.best_checkpoint_path,
        monitor='val_accuracy',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        period=cfg.best_checkpoint_freq
    )


def make_last_checkpointer(cfg):
    return ModelCheckpoint(
        filepath=cfg.last_checkpoint_path,
        verbose=0,
        save_weights_only=True,
        period=1
    )


def make_callbacks(cfg):
    if not os.path.exists(cfg.output_folder):
        os.makedirs(cfg.output_folder)

    callbacks = []

    if cfg.save_checkpoints:
        if not os.path.exists(cfg.checkpoint_folder):
            os.makedirs(cfg.checkpoint_folder)
        callbacks.append(make_best_checkpointer(cfg))
        callbacks.append(make_last_checkpointer(cfg))

    csv_path = os.path.join(cfg.output_folder, 'log.csv')
    csv_logger = CSVLogger(csv_path, append=(cfg.initial_epoch > 0))
    callbacks.append(csv_logger)

    if cfg.use_tensorboard_logger:
        logs_path = os.path.join(cfg.output_folder, 'logs')
        tensorboard_logger = TensorBoard(logs_path, write_graph=False, profile_batch=0)
        callbacks.append(tensorboard_logger)

    return callbacks


def train(cfg, callbacks=[]):
    initialize_session(cfg)
    model = make_model(cfg)
    data_loader = DataLoader(cfg)
    callbacks.extend(make_callbacks(cfg))

    train_dataset = data_loader.make_train_dataset(cfg.train_list)
    val_dataset = data_loader.make_test_dataset(cfg.val_list)
    result = model.fit(
        train_dataset,
        steps_per_epoch=cfg.n_batches,
        initial_epoch=cfg.initial_epoch,
        epochs=cfg.n_epochs,
        verbose=2,
        callbacks=callbacks,
        validation_data=val_dataset,
        validation_freq=cfg.val_freq
    )
    return result


def main():
    cfg = read_config()
    train(cfg)


if __name__ == '__main__':
    main()
