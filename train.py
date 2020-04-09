import os

import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam

from config import read_config
from data_loader import make_dataset
from sincnet import SincNetModel


def main():
    cfg = read_config()

    K.clear_session()

    model = SincNetModel(cfg)

    # optimizer = RMSprop(lr=cfg.lr, rho=0.9, epsilon=1e-8)
    optimizer = Adam(lr=cfg.lr)
    model.compile(
        loss=tfa.losses.TripletSemiHardLoss(),
        optimizer=optimizer,
        # metrics=['accuracy']
    )

    checkpoints_path = os.path.join(cfg.output_folder, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoints_path, 'SincNet-{epoch:04d}.hdf5'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        period=cfg.N_eval_epoch
    )

    logs_path = os.path.join(cfg.output_folder, 'logs')
    tensorboard_logger = TensorBoard(logs_path, write_graph=False)

    callbacks = [checkpointer, tensorboard_logger]

    if cfg.pt_file != 'none':
        model.load_weights(cfg.pt_file)

    train_dataset = make_dataset(cfg, cfg.train_list, for_train=True)
    validation_dataset = make_dataset(cfg, cfg.validation_list, for_train=False)
    model.fit(
        train_dataset,
        steps_per_epoch=cfg.N_batches,
        epochs=cfg.N_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_dataset,
        validation_steps=cfg.N_val_batches,
        validation_freq=cfg.N_eval_epoch
    )


if __name__ == '__main__':
    main()
