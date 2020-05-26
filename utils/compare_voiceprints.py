import sys

import numpy as np


path1 = sys.argv[1]
path2 = sys.argv[2]


def read_voiceprint(path):
    with open(path, 'r') as f:
        line = f.readline()
        voiceprint = list(map(float, line.split()))
        return np.array(voiceprint)


v1 = read_voiceprint(path1)
v2 = read_voiceprint(path2)


def dist(v1, v2):
    return 0.5 * (1 - v1.dot(v2))


print(dist(v1, v2))