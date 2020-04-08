import os

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop

from config import read_config
from data_loader import make_dataset
from sincnet import SincNetModel


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

    train_dataset = make_dataset(cfg, cfg.train_list, for_train=True)
    validation_dataset = make_dataset(cfg, cfg.validation_list, for_train=False)
    model.fit(
        train_dataset,
        steps_per_epoch=cfg.N_batches,
        epochs=cfg.N_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_dataset
    )


if __name__ == '__main__':
    main()
