import glob
import os

import cv2 as cv
from tqdm import tqdm

dataset = 'FLIR'

imgs = glob.glob(f'datasets/{dataset}/{dataset}_datasets/trainA/*')
# imgs = glob.glob('datasets/{dataset}/{dataset}_datasets/{dataset}_IR_edge_map/*')
path = f'datasets/{dataset}/{dataset}_datasets/{dataset}_VIS_edge_map/'

# os.mkdir(path, exist_ok=True)

for img in tqdm(imgs):
    name = img.split('/')[-1]
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"

    edges = cv.Canny(img, 255/3, 255)
    if not cv.imwrite(path + name, edges):
        raise Exception("Could not write image")
