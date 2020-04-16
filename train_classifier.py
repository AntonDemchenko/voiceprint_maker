from tensorflow.keras.optimizers import RMSprop

from config import read_config
from data_loader import ClassifierDataLoader
from sincnet import SincNetClassifierFactory
from training import train


def make_model(cfg):
    model = SincNetClassifierFactory(cfg).create()
    if cfg.pt_file != 'none':
        model.load_weights(cfg.pt_file)
    optimizer = RMSprop(lr=cfg.lr, rho=0.9, epsilon=1e-8)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


def main():
    cfg = read_config()
    model = make_model(cfg)
    data_loader = ClassifierDataLoader(cfg)
    train(cfg, model, data_loader)


if __name__ == '__main__':
    main()
