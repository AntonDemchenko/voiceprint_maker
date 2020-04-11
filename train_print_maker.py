import os

import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import RMSprop

from config import read_config
from data_loader import PrintMakerDataLoader
from sincnet import SincNetPrintMakerFactory


def main():
    cfg = read_config()

    K.clear_session()

    model = SincNetPrintMakerFactory(cfg).create()
    if cfg.pt_file != 'none':
        # Skip mismatch enables to load weights of networks with other head
        model.load_weights(cfg.pt_file, by_name=True, skip_mismatch=True)
    for layer in model.layers[:-2]:
        layer.trainable = False

    optimizer = RMSprop(lr=cfg.lr, rho=0.9, epsilon=1e-8)
    model.compile(
        loss=tfa.losses.TripletSemiHardLoss(),
        optimizer=optimizer,
    )

    checkpoints_path = os.path.join(cfg.output_folder, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoints_path, cfg.checkpoint_name),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        period=cfg.N_eval_epoch
    )

    logs_path = os.path.join(cfg.output_folder, 'logs')
    tensorboard_logger = TensorBoard(logs_path, write_graph=False)

    csv_path = os.path.join(cfg.output_folder, 'log.csv')
    csv_logger = CSVLogger(csv_path, append=(cfg.initial_epoch > 0))

    callbacks = [checkpointer, tensorboard_logger, csv_logger]

    data_loader = PrintMakerDataLoader(cfg)
    train_dataset = data_loader.make_train_dataset(cfg.train_list)
    validation_dataset = data_loader.make_test_dataset(cfg.validation_list)
    model.fit(
        train_dataset,
        steps_per_epoch=cfg.N_batches,
        initial_epoch=cfg.initial_epoch,
        epochs=cfg.N_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_dataset,
        validation_freq=cfg.N_eval_epoch
    )


if __name__ == '__main__':
    main()
