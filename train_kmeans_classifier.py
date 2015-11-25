import cv2
from sklearn.cluster import KMeans
import numpy as np
import pickle

image_ct = 60
descriptors = []
test = []

sift = cv2.xfeatures2d.SIFT_create()

for i in range(image_ct):
	imname = "dataset/img{:0>5d}.png".format(i)
	image = cv2.imread(imname)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	kp, des = sift.detectAndCompute(gray, None)
	descriptors.extend(des)

print len(descriptors)
print len(descriptors[0])

y_pred = KMeans(n_clusters=1000, max_iter=300).fit(descriptors)
f = open('kmeans_1000_300.dat', 'wb')
f.write(pickle.dumps(y_pred, protocol=2))
f.close()

print 'y'


'''
f = f = open('kmeans.dat', 'rb')
a = f.read()
f.close()

a = pickle.loads(a)
print 'tic'
a = a.predict(test)
print 'toc'
print a


image = cv2.imread("snap5.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
gray = cv2.Canny(gray,100,200)
sift = cv2.xfeatures2d.SIFT_create()

kp, des = sift.detectAndCompute(gray, None)

gray = cv2.drawKeypoints(gray, kp, gray)

print("#kps: {}, descriptors: {}".format(len(kp), des.shape))
cv2.imwrite('test_edged.png', gray)

kp, des = sift.detectAndCompute(gray, None)
img = gray.copy()
#img = cv2.drawKeypoints(img, kp, img)
#print("#kps: {}, descriptors: {}".format(len(kp), des.shape))
cv2.imwrite('test.png', img)

def dump(obj):
    for attr in dir(obj):
        print "obj.%s = %s" % (attr, getattr(obj, attr))


for i in range(10):
	
	img = gray.copy()
	img = cv2.bilateralFilter(img, 11, 17, 17)
	img = cv2.Canny(img,100,200)

	
	kp, des = sift.detectAndCompute(img, None)


	img = cv2.drawKeypoints(gray, kp, img)
	#img = cv2.drawKeypoints(filtered,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	#print("# i: {}, kps: {}, descriptors: {}".format(i, len(kp), des.shape))
	cv2.imwrite('test{}.png'.format(i), img)


print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
# kps: 274, descriptors: (274, 128)
surf = cv2.xfeatures2d.SURF_create()
(kps, descs) = surf.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
'''