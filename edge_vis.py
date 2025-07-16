import glob
import cv2 as cv


# imgs = glob.glob('datasets/FLIR/FLIR_datasets/trainC/*.jpeg')
imgs = glob.glob('datasets/FLIR/FLIR_datasets/FLIR_IR_edge_map/*.jpeg')
path = 'datasets/FLIR/FLIR_datasets/FLIR_IR_edge_map/'
# path = 'datasets/FLIR/FLIR_datasets/FLIR_VIS_edge_map/'

for img in imgs:
    name = img.split('/')[-1]
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"

    # edges = cv.Canny(img, 255/3, 255)
    cv.imwrite(path + name, 255 - img)
