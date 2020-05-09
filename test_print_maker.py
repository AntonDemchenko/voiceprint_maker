import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import read_config
from data_loader import DataLoader
from sincnet import create_print_maker


def check_norms(vectors):
    norms = [np.linalg.norm(v) for v in vectors]
    assert abs(1 - min(norms)) < 1e-6
    assert abs(1 - max(norms)) < 1e-6


def make_window_predictions(model, dataset):
    paths = []
    predictions = []
    for path_batch, signal_batch, _ in tqdm(dataset):
        prediction_batch = model.predict(signal_batch)
        paths.extend(path_batch)
        predictions.extend(prediction_batch)
    check_norms(predictions)
    return paths, predictions


def make_path_predictions_from_window_predictions(paths, predictions):
    path_to_embedding = dict()
    path_to_chunk_cnt = dict()
    for path, prediction in zip(paths, predictions):
        if path not in path_to_embedding:
            path_to_chunk_cnt[path] = 0
            path_to_embedding[path] = np.zeros(prediction.shape)
        path_to_embedding[path] += prediction
        path_to_chunk_cnt[path] += 1
    for path in path_to_embedding.keys():
        path_to_embedding[path] /= path_to_chunk_cnt[path]
    paths = [path for path, _ in path_to_embedding.items()]
    embeddings = [embedding for _, embedding in path_to_embedding.items()]
    embeddings = tf.math.l2_normalize(embeddings, axis=1).numpy()
    check_norms(embeddings)
    return paths, embeddings


def make_path_predictions(model, dataset):
    paths, predictions = make_window_predictions(model, dataset)
    return make_path_predictions_from_window_predictions(paths, predictions)


def distance(p1, p2):
    return 1 - p1.dot(p2)


def test(cfg, model, dataset):
    paths, predictions = make_path_predictions(model, dataset)
    assert len(paths) == len(predictions)

    labels = [cfg.path_to_label[path] for path in paths]
    label_to_cnt = dict()
    for l in labels:
        if l not in label_to_cnt:
            label_to_cnt[l] = 0
        label_to_cnt[l] += 1
    assert all(c == 2 for c in label_to_cnt.values())

    predicted_labels = []
    for i in range(len(predictions)):
        closest = None
        for k in filter(lambda k: k != i, range(len(predictions))):
            if closest is None or distance(predictions[i], predictions[k]) < distance(predictions[i], predictions[closest]):
                closest = k
        predicted_labels.append(labels[closest])

    correct_cnt = 0
    for expected, predicted in zip(labels, predicted_labels):
        if expected == predicted:
            correct_cnt += 1
    accuracy = correct_cnt / len(labels)
    return accuracy


def main():
    cfg = read_config()
    model = create_print_maker(cfg)
    # Skip mismatch enables to load weights of networks with other head
    model.load_weights(cfg.checkpoint_file, by_name=True, skip_mismatch=True)
    for layer in model.layers:
        layer.trainable = False
    dataset = DataLoader(cfg).make_test_iterable(cfg.test_list)
    accuracy = test(cfg, model, dataset)
    print(accuracy)


if __name__ == '__main__':
    main()
