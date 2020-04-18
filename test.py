import numpy as np
from tqdm import tqdm

from config import read_config
from data_loader import ClassifierDataLoader
from sincnet import SincNetClassifierFactory


def test(cfg, model, data_loader, path_list):
    dataset = data_loader.make_test_dataset(path_list)
    path_to_prediction_sum = dict()
    for path_batch, signal_batch in tqdm(dataset.as_numpy_iterator()):
        prediction_batch = model.predict(signal_batch)
        for path, prediction in zip(path_batch, prediction_batch):
            path = path.decode()
            if path not in path_to_prediction_sum:
                path_to_prediction_sum[path] = np.zeros([cfg.n_classes])
            path_to_prediction_sum[path] += prediction
    error_count = 0
    for path in path_list:
        prediction_sum = path_to_prediction_sum[path]
        expected_label = cfg.lab_dict[path]
        predicted_label = np.argmax(prediction_sum)
        if expected_label != predicted_label:
            error_count += 1
    accuracy = 1 - (error_count / len(path_list))
    return accuracy


def main():
    cfg = read_config()
    model = SincNetClassifierFactory(cfg).create()
    data_loader = ClassifierDataLoader(cfg)
    accuracy = test(cfg, model, data_loader, cfg.test_list)
    print(accuracy)


if __name__ == '__main__':
    main()
