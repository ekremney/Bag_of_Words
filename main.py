from train import *
from tools import *
from libsvm.svmutil import svm_train, svm_save_model
from ml_ops import search, train, bootstrap
import argparse, sys, datetime

modes = ['train', 'bootstrap', 'search']
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="specify the mode [-m train|bootstrap|search]", required=True, choices=modes)
args = parser.parse_args()

if args.mode == 'train':
	hists, labels = train()
	
	svm = svm_train(labels, hists, '-s 0 -t 0 -c 1')

	model_name = 'svm_trained_' + str(datetime.datetime.now()).replace(' ', '_') + '.dat'
	svm_save_model(model_name, svm)


if args.mode == 'bootstrap':
	hists, labels = train()
	hists, labels = bootstrap(hists, labels)
	
	svm = svm_train(labels, hists, '-s 0 -t 0 -c 1')

	model_name = 'svm_bootstrapped_' + str(datetime.datetime.now()).replace(' ', '_') + '.dat'
	svm_save_model(model_name, svm)

if args.mode == 'search':
	search()

