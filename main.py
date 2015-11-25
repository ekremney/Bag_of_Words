from train import *
from tools import *
from sw_search import *
import cv2

for i in range (0,10):
	img = img_read(folder, i)
	img, detections = initialize(img)
	cv2.imwrite(str(i) + '.png', img)





