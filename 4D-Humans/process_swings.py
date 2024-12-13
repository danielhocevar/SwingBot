from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import pandas as pd
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

data_dir = Path('/home/daniel/development/csc2626/final-project/golfdb/data/')

df = pd.read_csv(os.path.join(data_dir, 'golfDB_processed.csv'))

tiger_swings = df[df['player'] == 'TIGER WOODS']

print(tiger_swings)

import subprocess

for idx, row in tiger_swings.iterrows():
    print('Running pose detection on', row['processed_video_path'])
    subprocess.run([
        'python', 'track.py',
        f'video.source=/home/daniel/development/csc2626/final-project/golfdb/data/{row["processed_video_path"]}',
        f'video.output_dir=/home/daniel/development/csc2626/final-project/golfdb/data/poses/'
    ])
