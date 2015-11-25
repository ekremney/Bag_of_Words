from tools import *
from imageops import *
from libsvm.svmutil import svm_predict
import time

def sliding_window_search(img, sbox_height, sbox_width, threshold, _svm = None):
	detections = []
	height, width = img.shape

	if _svm  is not None:
		svm = _svm

	for i in range(1, (height - sbox_height), slide):
		for j in range(1, (width - sbox_width), slide):
			img_patch = img[i:i+sbox_height-1, j:j+sbox_width-1]
			img_feat = extract(img_patch)
			vw_hist = calculate_visual_word(img_feat)

			plabel, acc, pr = svm_predict([0], [vw_hist], svm)
			#time.sleep(10000)
			
			if pr[0][0] > threshold:
				detections.append([i, i+sbox_height-1, j, j+sbox_width-1, pr[0][0]])
	return detections

def initialize(img, _svm = None):
	detections = []

	height, width = img.shape	
	
	for i in s_windows:
		if _svm is not None:
			d = sliding_window_search(img, i[0], i[1], i[2], _svm)
		else:
			d = sliding_window_search(img, i[0], i[1], i[2])
		d = add_bbox_margin(d, height, width, sign = -1)
		[detections.append(j) for j in d]

	detections = non_max_suppression(detections, non_max_thresh)

	for j in detections:
		img = cv2.rectangle(img,(j[2],j[0]),(j[3],j[1]),(0,255,0),1)
		cv2.putText(img, str(j[4])[:5], (int(j[2]),int(j[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255))

	return img, detections