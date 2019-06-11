import os
import numpy as np
import operator
import time

import pickle
import glob
import random
import yaml
import itertools
import copy # we use it to do copy of list, dictionary, rather than just create a reference
from sklearn.utils import shuffle

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt




def ensure_folder_exists(folder_path):
#The folder_path should be an absolute directory
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def batch_convert_prob_to_onehot(pred_mtx):
	# This function is used to convert a group of softmaxed vectors to a group of one-hot vectors
	# The prediction matrix should have the shape [num_of_vectors, vector_size]
	num_of_vectors=pred_mtx.shape[0]
	vector_size=pred_mtx.shape[1]
	for i,v in enumerate(pred_mtx):
		max_idx, value = max(enumerate(v), key=operator.itemgetter(1))
		onehot=np.zeros(vector_size)
		onehot[max_idx]=1
		pred_mtx[i]=onehot


def numeric_stable_std(std_vec):		#Convert input std vector to a numerically stable one
	# Replacing Not a Number (NaN) with zero, (positive) infinity with a very large number and negative infinity with a very small (or negative) number.
	vec = np.nan_to_num(std_vec) 
	# Replace 0s in the vector with 1s
	for idx, ele in enumerate(vec):
		if ele == 0: 
			vec[idx] = 1
	
	return vec
	

def feature_segmentation(feat_mtx, label, segment_length, hop_size, num_of_segments):
	# INPUTS:
	# 1. feat_mtx:							have the dimension [num_of_files, num_of_frames, feature_dimension]
	# 2. label:								have the dimension [num_of_flies,(optional: by default the label is a scalar, but it can also be in the form of one-hot vector)]
	# OUTPUTS:
	# 1. feat_mtx_segmented:		have the dimension [num_of_files, num_of_segments, segment_length, feature_dimension]
	# 2. label_segmented:				have the dimension [num_of_files, num_of_segments, (optional)]
	
	feat_mtx_segmented=[]
	for file_idx, file_feat in enumerate(feat_mtx):
		file_feat_segments=[]
		for segment_idx in range(num_of_segments):	#after this loop, file_feat_segments is [num_of_segments, segment_length, feature_dimension]
			feat_segment=file_feat[segment_idx * hop_size: (segment_idx * hop_size) + segment_length, :]
			file_feat_segments.append(feat_segment)
		feat_mtx_segmented.append(file_feat_segments)
		
	#Convert list to a numpy array
	feat_mtx_segmented=np.array(feat_mtx_segmented)
	
	label_segmented=[]
	for ele in label:
		file_labels = []
		for segment_idx in range(num_of_segments):
			file_labels.append(ele)
		label_segmented.append(file_labels)
			
	#Convert list to a numpy array
	label_segmented=np.array(label_segmented)
	
	return feat_mtx_segmented, label_segmented
	


def plotLearningCurve(train_loss, test_loss, n_steps, result_folder_name='result'):
	# Input:
	#		train_loss: a list contains the training set loss at different time
	#		test_loss: a list contains the validation set loss at different time
	#		n_steps: the total number of training steps
	# Output: None
	
	if (len(train_loss) != len(test_loss)):
		print("Error when plotting the curve: the train_loss and test_loss should have the same length.")
		return
	
	num_of_records = len(train_loss)
	title = "Learning Curves"
	plt.title(title)
	plt.xlabel("Training Steps")
	plt.ylabel("Loss")
	plt.grid()
	
	base = np.linspace(0, n_steps, num_of_records)
	plt.plot(base, train_loss, 'o-', color = 'r', label="Training Loss") 
	plt.plot(base, test_loss, 'o-', color = 'b', label="Validation Loss") 
	plt.legend(loc="best")
	#plt.show(block=False)
	ensure_folder_exists(result_folder_name)
	plt.savefig(result_folder_name + '/learning_curve.png')
	


def segmentResultAveraging(model_output, num_of_segments):
	n_total_segments = model_output.shape[0]
	n_samples = n_total_segments / num_of_segments
	
	# Generate averaged result matrix
	results = []
	for i in range(n_samples):
		result_single = np.mean(model_output[i * num_of_segments : (i + 1) * num_of_segments], axis = 0)
		results.append(result_single)
		
	#Convert list to numpy array
	results = np.array(results)
	return results
	
def getConfusionMatrix(pred_labels, true_labels, n_classes):
	# the confusion matrix has y-axis being true classes, x-axis being predictions.
	cnf_mtx = np.zeros([n_classes,n_classes])
	for i in range(len(true_labels)):
		cnf_mtx[true_labels[i]][pred_labels[i]] += 1
	cnf_mtx = cnf_mtx.astype('int')
	return cnf_mtx
	
def plot_confusion_matrix(cm, classes,normalize=False, savefig_name = 'cnf_mtx.png', save_folder_name = '', title='Confusion matrix',cmap=plt.cm.Blues):
	# This function prints and plots the confusion matrix.
	# Normalization can be applied by setting `normalize=True`.
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
	print(cm)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	#plt.show(block=False)
	ensure_folder_exists(os.getcwd() + '/' + save_folder_name)
	plt.savefig(save_folder_name + '/' + savefig_name)
	

# The tic toc functions are originally copied from the following source, though they are modified a little bit.
# Ref: https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python 	
def tic():
	#Homemade version of matlab tic and toc functions
	global startTime_for_tictoc
	startTime_for_tictoc = time.time()
	
def toc():
	if 'startTime_for_tictoc' in globals():
		print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
		del globals()['startTime_for_tictoc']
	else:
		print("Toc: start time not set")

