from tools import *
from imageops import *
from random import sample
from libsvm.svmutil import svm_predict
import time

def train():
	hists = []
	p_feat = 0
	labels = []

	print 'Extracting positive features...'

	for i in train_indexes:
		img = img_read(folder, i)

		h, w = img.shape
		bboxes = add_bbox_margin(read_bboxes(folder, i), h, w)

		for j in bboxes:
			img_cut = img[j[0]:j[1], j[2]:j[3]]

			feat = extract(img_cut)
			if feat is not None:
				hist = calculate_visual_word(feat)
				hists.append(hist)
				p_feat += 1
				labels.append(1)

	print 'Extracting negative features...'

	index = 0
	while index < p_feat * neg_weight:
		i = sample(train_indexes, 1)[0]

		img = img_read(folder, i)

		h, w = img.shape
		bboxes = add_bbox_margin(read_bboxes(folder, i), h, w)
		neg_bb = rand_bbox(bboxes, h, w)

		if overlaps(neg_bb, bboxes, 0.01) != -1:
			continue

		img_cut = img[neg_bb[0]:neg_bb[1], neg_bb[2]:neg_bb[3]]
		feat = extract(img_cut)

		if feat is not None:
			hist = calculate_visual_word(feat)
			hists.append(hist)
			labels.append(-1)
			index += 1

	return hists, labels


def bootstrap(hists, labels):

    print "Starting bootstrapping..."

    # Bootstrapping
    for i in bootstrap_indexes:
        success = False
        while not success:
            img = img_read(folder, i)
            bboxes = read_bboxes(folder, i)

            detections = initialize_sw(img)

            hard_negatives = []
            for j in detections:
                if overlaps(j, bboxes) == -1:
                    hard_negatives.append(j)
            height, width = img.shape
            hard_negatives = add_bbox_margin(hard_negatives, height, width)

            for j in hard_negatives:
                img_cut = img[j[0]:j[1], j[2]:j[3]]
                feat = extract(img_cut)

                if feat is not None:
                    hist = calculate_visual_word(feat)
                    hists.append(hist)
                    labels.append(-1)
                    success = True
    
    return hists, labels

def search():

	print 'Sliding search starting...'

	for i in test_indexes:
		img = img_read(folder, i)

		detections = initialize_sw(img)
		detections = non_max_suppression(detections, non_max_thresh)

		for j in detections:
			img = cv2.rectangle(img,(j[2],j[0]),(j[3],j[1]),(0,255,0),1)
			cv2.putText(img, str(j[4])[:5], (int(j[2]),int(j[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255))

		filename = "detection{:0>5d}.png".format(i)
		cv2.imwrite(filename, img)
		



