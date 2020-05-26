import os
import sys
import wave

path_prefix = sys.argv[1]
path_list_file = sys.argv[2]

with open(path_list_file, 'r') as f:
    path_list = list(map(str.strip, f.readlines()))

durs = []
for path in path_list:
    path = os.path.join(path_prefix, path)
    audio = wave.open(path, 'rb')
    d = audio.getnframes() / audio.getframerate()
    durs.append(d)

durs.sort()

print('Min', min(durs), 'sec')
print('Median', durs[len(durs) // 2], 'sec')
print('Max', max(durs), 'sec')
print('Mean', sum(durs) / len(durs), 'sec')
print('Sum', sum(durs) / 3600, 'hour(s)')