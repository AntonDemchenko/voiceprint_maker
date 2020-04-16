import tensorflow_addons as tfa
from tensorflow.keras.optimizers import RMSprop

from config import read_config
from data_loader import PrintMakerDataLoader
from sincnet import SincNetPrintMakerFactory
from training import train


def make_model(cfg):
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
    return model


def main():
    cfg = read_config()
    model = make_model(cfg)
    data_loader = PrintMakerDataLoader(cfg)
    train(cfg, model, data_loader)


if __name__ == '__main__':
    main()
