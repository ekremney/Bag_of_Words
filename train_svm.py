from train import *
from bootstrap import *
from libsvm.svmutil import svm_train, svm_save_model

hists, labels = train(selected_indexes)

svm = svm_train(labels, hists, '-s 0 -t 0 -c 1')

hists, labels = bootstrap(hists, labels, svm)
svm = svm_train(labels, hists, '-s 0 -t 0 -c 1')
svm_save_model('svm.dat', svm)
