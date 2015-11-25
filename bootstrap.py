from tools import *
from sw_search import *
from imageops import *

def bootstrap(hists, labels, _svm):

    print "Starting bootstrapping..."

    # Bootstrapping
    for i in bootstrap_indexes:
        img = img_read(folder, i)
        bboxes = read_bboxes(folder, i)

        _, detections = initialize(img, _svm = _svm)

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
    
    return hists, labels