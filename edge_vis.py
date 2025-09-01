import glob
import os

import cv2 as cv
from tqdm import tqdm

imgs = glob.glob('datasets/LYNRED/LYNRED_datasets/trainA/*.jpg')
# imgs = glob.glob('datasets/LYNRED/LYNRED_datasets/LYNRED_IR_edge_map/*.jpg')
path = 'datasets/LYNRED/LYNRED_datasets/LYNRED_VIS_edge_map/'

# os.mkdir(path, exist_ok=True)

for img in tqdm(imgs):
    name = img.split('/')[-1]
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"

    edges = cv.Canny(img, 255/3, 255)
    if not cv.imwrite(path + name, edges):
        raise Exception("Could not write image")
