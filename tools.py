from ConfigParser import ConfigParser
import pickle
from libsvm.svmutil import svm_load_model

def get_indexes(kind):
	train_str = cfg.get('indexes', kind)
	indexes = eval(train_str)
	result = [j for i in indexes for j in range(i[0], i[2], i[1])]
	return sorted(result)

cfg = ConfigParser()
cfg.read('config.ini')

f = open(cfg.get('kmeans', 'path'), 'rb')
kmeans = pickle.loads(f.read())
f.close()

train_indexes = get_indexes('train')
bootstrap_indexes = get_indexes('bootstrap')
test_indexes = get_indexes('test')

slide = cfg.getint('params', 'slide')
folder = cfg.get('params', 'folder')
neg_weight = cfg.getint('params', 'neg_weight')
marginX = cfg.getint('params', 'marginX')
marginY = cfg.getint('params', 'marginY')
non_max_thresh = cfg.getfloat('params', 'non_max_threshold')

get_c_iteration = cfg.getboolean('svm', 'c_iteration')
def_c_value = cfg.getfloat('svm', 'def_c_value')
s_windows = eval(cfg.get('svm', 's_windows'))
svm = svm_load_model(cfg.get('svm', 'path'))

mode = cfg.get('configs', 'mode')



print bootstrap_indexes

