import cv2, csv
import numpy as np
from tools import *
from random import sample, randint
from libsvm.svmutil import svm_predict

def img_read(folder, index):
	filename = folder + "/img{:0>5d}.png".format(index)
	return cv2.imread(filename, 0)

def read_bboxes(folder, index):
	result = []
	annot_filename = folder + "/img{:0>5d}.annot".format(index)

	with open(annot_filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		i = 0
		for row in reader:
			result.append([])
			for element in row:
				result[i].append(int(element))
			i += 1
	return result

def add_bbox_margin(bboxes, height, width, sign = 1):
	if bboxes is None:
		return
	for row in bboxes:
		row[0] = max(1, row[0] - (marginY*sign))
		row[1] = min(height, row[1] + (marginY*sign))
		row[2] = max(1, row[2] - (marginX*sign))
		row[3] = min(width, row[3] + (marginX)*sign)
	return bboxes

def rand_bbox(bboxes, row, col):
	bb = sample(bboxes, 1)[0]
	height = bb[1] - bb[0] + 1
	width = bb[3] - bb[2] + 1
	maxY = row - height
	maxX = col - width
	y = randint(0, maxY-1)
	x = randint(0, maxX-1)
	neg_bb = [y, y+height-1, x, x+width-1]
	return neg_bb

def overlaps(neg_bb, bboxes, th=0.25):
	does_overlap = -1
	index = 0
	for i in bboxes:
		top = max(i[0], neg_bb[0])
		bottom = min(i[1], neg_bb[1])
		left = max(i[2], neg_bb[2])
		right = min(i[3], neg_bb[3])

		if bottom-top > 0 and right-left > 0:
			intersection = float((bottom-top)*(right-left))

			h1 = neg_bb[1] - neg_bb[0] + 1
			w1 = neg_bb[3] - neg_bb[3] + 1
			h2 = i[1] - i[0] + 1
			w2 = i[3] - i[2] + 1

			union = float(((h1*w1)+(h2*w2)) - intersection)

			if intersection/union > th:
				does_overlap = index
				break
		index += 1

	return does_overlap

#  Felzenszwalb et al.
def non_max_suppression(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1, x2, y1, y2, resemblance, areas = [], [], [], [], [], []
	for i in boxes:
		x1.append(i[2])
		x2.append(i[3])
		y1.append(i[0])
		y2.append(i[1])
		resemblance.append(i[4])
		areas.append((i[3]-i[2])*(i[1]-i[0]))

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	idxs = np.argsort(resemblance)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]

		# loop over all indexes in the indexes list
		for pos in xrange(0, last):
			# grab the current index
			j = idxs[pos]

			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])

			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)

			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / areas[j]

			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)

		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)

	# return only the bounding boxes that were picked
	res = []
	for i in pick:
		res.append(boxes[i])
	return res

def calculate_visual_words(des):
	'''
		This function takes a kmeans classifier and feature descriptors
		of an image. Calculates and returns visual word histograms
	'''
	h = [kmeans.predict(i) for i in des]

	hists = [[0]*1000 for i in range(len(des))]

	for i in range(len(h)):
		for j in range(len(h[i])):
			hists[i][j] =+ 1
	return hists

def calculate_visual_word(des):
	'''
		This function takes a kmeans classifier and feature descriptors
		of an image. Calculates and returns visual word histograms
	'''
	if des is None:
		return []
	h = kmeans.predict(des)
	hist = [0]*1000

	for i in h:
		hist[i] =+ 1
	return hist


def extract(img, filtered=None, edged=None):
	if filtered:
		img = cv2.bilateralFilter(img, 11, 17, 17)
	if edged:
		img = cv2.Canny(img,100,200)
	sift = cv2.xfeatures2d.SIFT_create()
	_, des = sift.detectAndCompute(img, None)
	return des


def sliding_window_search(img, sbox_height, sbox_width, threshold):
	detections = []
	height, width = img.shape

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


def initialize_sw(img):
	detections = []

	height, width = img.shape	
	
	for i in s_windows:
		d = sliding_window_search(img, i[0], i[1], i[2])
		d = add_bbox_margin(d, height, width, sign = -1)
		[detections.append(j) for j in d]

	return detections

def compute_detection_AP(detections, bboxes, th=0.4):
	# sort detections
	npos = len(bboxes)
	tp = [0] * len(detections)
	bb_used = [0] * len(bboxes)
	
	for i in range(len(detections)):
		k = overlaps(detections[i], bboxes, th)
		if k != -1 and bb_used[k]==0: # TODO: update
			tp[i] = 1
			bb_used[k] = 1;
			#del bboxes[k]
	pr = [0] * npos
	rc = [0] * npos
	j = 0
	detected_num = sum(tp)
	for i in range(len(tp)):
		if tp[i] == 1:
			pr[j] = float(sum(tp[:i+1]))/(i+1)
			rc[j] = float(j+1)/float(npos)
			j += 1
	[pr.append(0) for i in range(npos-detected_num)]
	[rc.append(rc[j-1]) for i in range(npos-detected_num)]
	
	ap = np.mean(pr)
	
	return ap, pr, rc
