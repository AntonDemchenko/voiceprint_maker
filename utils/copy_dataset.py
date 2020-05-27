import os
import shutil
import sys

from tqdm import tqdm

path_prefix = sys.argv[1]
path_list_file = sys.argv[2]
output_directory = sys.argv[3]

with open(path_list_file, 'r') as f:
    path_list = list(map(str.strip, f.readlines()))

for path in tqdm(path_list):
    source_path = os.path.join(path_prefix, path)
    target_path = os.path.join(output_directory, path)
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(source_path, target_path)
