from tools import *
from imageops import *
from random import sample
import time

def train(selected_indexes = None):
	hists = []
	p_feat = 0
	labels = []

	if selected_indexes is not None:
		train_indexes = selected_indexes

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