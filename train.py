from config import read_config
from data_loader import DataLoader
from sincnet import SincNetModelFactory
from training import make_optimizer
from training import train


def make_model(cfg):
    model = SincNetModelFactory(cfg).create()
    if cfg.checkpoint_file != 'none':
        model.load_weights(cfg.checkpoint_file)
    optimizer = make_optimizer(cfg)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


def main():
    cfg = read_config()
    model = make_model(cfg)
    data_loader = DataLoader(cfg)
    train(cfg, model, data_loader)


if __name__ == '__main__':
    main()
