import os

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import RMSprop

from config import read_config
from data_loader import make_dataset
from sincnet import SincNetModelFactory


def make_print_maker_from_classifier(cfg, classifier_model_path):
    classifier = SincNetModelFactory(cfg).create()
    classifier.load_weights(classifier_model_path)
    x = classifier.layers[-2].output
    x = layers.Dense(cfg.out_dim)(x)
    x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    print_maker = tf.keras.Model(inputs=classifier.input, outputs=x)
    for layer in print_maker.layers[:-2]:
        layer.trainable = False
    return print_maker


def main():
    cfg = read_config()

    K.clear_session()

    model = make_print_maker_from_classifier(cfg, cfg.pt_file)
    if cfg.print_maker_pt_file != 'none':
        model.load_weights(cfg.print_maker_pt_file)

    optimizer = RMSprop(lr=cfg.lr, rho=0.9, epsilon=1e-8)
    model.compile(
        loss=tfa.losses.TripletSemiHardLoss(),
        optimizer=optimizer,
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
