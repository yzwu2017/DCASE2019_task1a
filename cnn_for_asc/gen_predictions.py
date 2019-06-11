# Script for DCASE2019 task1-a evaluation
# Load the leaderboard data, and generate the csv file containing the model outputs (for submission to leaderboard).

import numpy as np
import pickle
import glob
import random
import yaml
import copy # we use it to do copy of list, dictionary, rather than just create a reference
from sklearn.utils import shuffle
import itertools
from src.functions_asc import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

#Load the configurations
with open('general_config.yaml','r') as f:
	config = yaml.load(f)

gpu = 0
trained_model_folders = [	'results-logmel128S-AlexNetS-Mixup-24eps-Model0',
							'results-logmel128S-AlexNetS-Mixup-24eps-Model1',
							'results-logmel128S-AlexNetS-Mixup-24eps-Model2',
							'results-logmel128S-AlexNetS-Mixup-24eps-Model3',
						]

prediction_results_folder = 'fused-results'

#==============================================
#Load Feature Data 
#==============================================
print('Loading Data and Feature Pre-processing. ')

## Define the classes corresponding label index
num_of_classes = config['class_spec']['asc2019']['number_of_classes']
label_dict=copy.copy(config['class_spec']['asc2019']['index_assignment'])

evaluation_data_path = '/home/yzwu/DCASE2019_task1a/features/leaderboard/logmel-128-HF11LF51'
evaluation_setup_file = '/home/yzwu/DCASE2019_task1a/datasets/TAU-urban-acoustic-scenes-2019-leaderboard/evaluation_setup/test.csv'
file_extension = 'logmel'
FEATURE_DIMENSION = 128



def load_evaluation_data_3c(evaluation_data_path, eval_sample_namelist, file_extension):
	# Inspect the data folder
	sample_paths = glob.glob(evaluation_data_path + '/*.' + file_extension) # Inspect the training folder, and get the full paths of all the training samples
	sample_names=[]
	for ele in sample_paths:
		sample_names.append(ele.split("/")[-1].split('.')[0]) # From the training samples' full paths, generate the filenames list (without extension)
	# Read the evaluation samples' namelist.
	with open(evaluation_setup_file, "r") as text_file:
		lines = text_file.read().split('\n')
	for idx, ele in enumerate(lines):
		lines[idx]=lines[idx].split('/')[-1].split('.')[0]
		lines[idx]=lines[idx].split('\r')[0]
	lines = [ele for ele in lines if ele != ''] # only keep the non-empty elements
	lines = lines[1:] # remove header
	eval_sample_namelist=np.array(lines)
	del lines[:]
	## Read the feature matrix for each audio file (based on the namelist)
	feat_mtx=[]
	for filename in eval_sample_namelist:
		filepath = evaluation_data_path + '/' + filename + '.' + file_extension
		with open(filepath,'rb') as f:
			temp=pickle.load(f, encoding='latin1')
			feat_mtx.append(temp['feat_lmh'])
	#feat_mtx=np.array(feat_mtx)
	## pad the audio sequences with vectors with low value (representing silence), to make the sequence length a multiple of pad_unit.
	pad_unit = 128
	low_energy_value = -80 #the minimum value for this dataset is -96.13
	for idx, feat in enumerate(feat_mtx):
		feat_len = feat.shape[-2]
		if feat_len % pad_unit != 0:
			n_pad = pad_unit - feat_len % pad_unit
			pad_mtx = low_energy_value * np.ones([3,n_pad,FEATURE_DIMENSION]) 
			feat_padded = np.concatenate((feat, pad_mtx),axis = -2)
			feat_mtx[idx] = feat_padded
	feat_mtx = np.array(feat_mtx)
	return feat_mtx, eval_sample_namelist

feat_mtx_eval, eval_sample_namelist = load_evaluation_data_3c(evaluation_data_path, evaluation_setup_file, file_extension)



#==============================================
#Feature Segmentation
#==============================================
# In this code block, we will cut the feature of each audio file into segments (time slices).

hop_size = 128			# Default is 128
segment_length = 128	# Default is 128
def feature_segmentation_3c(feat_mtx, label_info, hop_size, segment_length):
	feat_mtx_segmented = []
	label_info_segmented = []
	for idx, feat in enumerate(feat_mtx):
		audio_length = feat.shape[-2]
		n_segments = ((audio_length - segment_length) // hop_size) + 1
		audio_segs = np.empty([n_segments,segment_length,FEATURE_DIMENSION])
		# Create segments of data
		for segment_idx in range(n_segments):	#after this loop, file_feat_segments is [num_of_segments, segment_length, feature_dimension]
			seg_data = feat[:, segment_idx * hop_size: (segment_idx * hop_size) + segment_length, :]
			feat_mtx_segmented.append(seg_data)
		# Create corresponding labels
		for segment_idx in range(n_segments):	#after this loop, file_feat_segments is [num_of_segments, segment_length, feature_dimension]
			label_info_segmented.append(label_info[idx])
	# Convert to numpy array
	feat_mtx_segmented = np.array(feat_mtx_segmented)
	label_info_segmented = np.array(label_info_segmented)
	return feat_mtx_segmented, label_info_segmented

feat_eval_seg, eval_seg_namelist = feature_segmentation_3c(feat_mtx_eval, eval_sample_namelist, hop_size, segment_length)



for mdl_idx, trained_model_folder in enumerate(trained_model_folders):
	print('Evaluating Model ' + str(mdl_idx) + '...')
	#==============================================
	#Feature Normalization
	#==============================================
	# In this code block, we will normalize the data to zero mean and unit variance

	eval_x = feat_eval_seg

	normalization_stats = pickle.load(open(trained_model_folder+'/normstats.pickle','rb'))	
	mean_train = normalization_stats['mean_train']
	std_train = normalization_stats['std_train']

	# Do feature normalization (zero mean and unit variance)
	eval_x = (eval_x - mean_train) / std_train

	#==============================================
	#Model Structure
	#==============================================
	print('Initialize Model Structure. ')


	class AlexNetS(nn.Module):
		def __init__(self):
			super(AlexNetS, self).__init__()
			def conv_bn(inp, oup, kernel, stride, pad, groups):
				return nn.Sequential(
					nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=kernel, stride=stride, padding=pad, groups=groups, bias=False),
					nn.BatchNorm2d(oup),
					nn.ReLU(inplace=True)
				)
			self.classifier = nn.Sequential(
				nn.Linear(192 * 8 * 8, 1024),
				nn.BatchNorm1d(1024),
				nn.ReLU(inplace=True),
				nn.Linear(1024, 256),
				nn.BatchNorm1d(256),
				nn.ReLU(inplace=True),
				nn.Linear(256, num_of_classes, bias=True),
			)
			self.features = nn.Sequential(
				conv_bn(inp=3,oup=48,kernel=3,stride=1,pad=1,groups=3),
				nn.MaxPool2d(kernel_size=2, stride=2),
				conv_bn(inp=48,oup=96,kernel=3,stride=1,pad=1,groups=3),
				nn.MaxPool2d(kernel_size=2, stride=2),
				conv_bn(inp=96,oup=192,kernel=3,stride=1,pad=1,groups=3),
				nn.MaxPool2d(kernel_size=2, stride=2),
				conv_bn(inp=192,oup=192,kernel=3,stride=1,pad=1,groups=3),
				conv_bn(inp=192,oup=192,kernel=3,stride=1,pad=1,groups=3), # For (100,128) input, the feature map size after this layer is (24,31)
				nn.MaxPool2d(kernel_size=2, stride=2), #If this is used, then feature map size =(11,15)
				#nn.AdaptiveAvgPool2d((1,1)),
			)
		def forward(self, x):
			x = self.features(x)
			x = x.view(x.size(0), -1)
			x = self.classifier(x)
			return x


	myModel = AlexNetS().cuda(gpu)
	myModel.load_state_dict(torch.load(trained_model_folder+'/myModel.dict'))


	BATCH_SIZE = 100

	#==============================================
	# Feed the Evaluation Data and Obtain Model Outputs
	#==============================================	
	print('Feeding Evaluation Data into the Model... ')

	eval_x = torch.Tensor(eval_x)

	myModel.eval()
	n_eval = eval_x.shape[0]
	# Feed the evaluation data into the model
	eval_outputs_seg = np.empty([n_eval, num_of_classes]) # Store the model outputs. 
	batch_indices = torch.Tensor(np.arange(BATCH_SIZE)).long()
	while batch_indices[0] < n_eval:
		# Get a batch of training data and label
		if batch_indices[0] +  BATCH_SIZE <= n_eval:
			# If the indices are all valid, we directly use them.
			batch_x = eval_x[batch_indices].cuda(gpu)
			# Specify the indices of model outputs for this batch
			eval_outputs_indices = batch_indices.cpu().numpy()
			# Update indices
			batch_indices = batch_indices + BATCH_SIZE
		else:
			# If the indices are out of range, we need to discard some of them.
			num_remain_audio = n_eval - batch_indices[0]
			batch_indices = batch_indices[0 : num_remain_audio]
			# Use the valid indices to get the batch data and label.
			batch_x = eval_x[batch_indices]
			# Specify the indices of model outputs for this batch
			eval_outputs_indices = batch_indices.cpu().numpy()
			# Update indices
			batch_indices = batch_indices + num_remain_audio
			
		# Make the batch data dimension match the input spec. of nn.Conv2D
		with torch.no_grad():
			#batch_x = batch_x.unsqueeze(1)
			batch_x = Variable(batch_x).cuda(gpu)
			# Get the model output given the test batch
			eval_out = myModel(batch_x)
			eval_out = torch.sigmoid(eval_out) # This is the model output
		eval_outputs_seg[eval_outputs_indices] = eval_out.cpu().data.numpy()

	## Obtain the sample-level output vector from segment-level output vectors.
	eval_outputs = [] # the model outputs.
	eval_files = []
	current_file = None
	for idx, fname in enumerate(eval_seg_namelist):
		if current_file == None: # If we are dealing with the first segment.
			current_file = fname
			output_avg_seg = eval_outputs_seg[idx]
			count_of_seg = 1
		elif current_file == fname: # If this segment belongs to the same audio sample as the previous segment.
			output_avg_seg = output_avg_seg + eval_outputs_seg[idx]
			count_of_seg = count_of_seg + 1
		elif current_file != fname: # If this segment belongs to a new audio sample
			# We have obtain all segments for the last audio sample,so save it.
			output_avg_seg = output_avg_seg / float(count_of_seg)
			eval_outputs.append(output_avg_seg)
			eval_files.append(current_file)
			# Update the current file to the new file.
			current_file = fname
			output_avg_seg = eval_outputs_seg[idx]
			count_of_seg = 1
	# save the averaged output vector for the last audio sample.
	output_avg_seg = output_avg_seg / float(count_of_seg)
	eval_outputs.append(output_avg_seg)
	eval_files.append(current_file)
	# Convert list to numpy array.
	eval_outputs = np.array(eval_outputs)
	eval_files = np.array(eval_files)


	## Obtain predictions
	# Convert outputs from one-hot vector to labels.
	predictions = 99 * np.ones(len(eval_outputs))
	for i in range(len(eval_outputs)):
		predictions[i] = np.argmax(eval_outputs[i])
	predictions = predictions.astype('int')
	class_names = np.array(['airport','shopping_mall','metro_station','street_pedestrian','public_square','street_traffic','tram','bus','metro','park'])


	#==============================================
	# Save the model predictions as a CSV file
	#==============================================	
	f = open(trained_model_folder + '/predictions_on_eval.csv','w')
	f.write('Id,Scene_label\n')
	for i, class_id in enumerate(predictions):
		f.write(eval_files[i]+','+class_names[class_id]+'\n') #Give your csv text here.
	f.close()

	pickle.dump(predictions,open(trained_model_folder + '/predictions_on_eval.pickle','wb'))
	pickle.dump(eval_outputs,open(trained_model_folder + '/predictions_soft_on_eval.pickle','wb'))



print('Save the Fused Result. ')
ensure_folder_exists(prediction_results_folder)
# Model Prediction Fusion
preds_soft=[]
for model_folder in trained_model_folders:
	preds_soft.append(pickle.load(open(model_folder + '/predictions_soft_on_eval.pickle','rb')))

preds_soft_fused = np.zeros(preds_soft[0].shape)
for ele in preds_soft:
	preds_soft_fused = preds_soft_fused + ele
preds_soft_fused = preds_soft_fused / float(len(trained_model_folders))


# Convert outputs from one-hot vector to labels.
preds_fused = 99 * np.ones(len(preds_soft_fused))
for i in range(len(preds_soft_fused)):
	preds_fused[i] = np.argmax(preds_soft_fused[i])
preds_fused = preds_fused.astype('int')
class_names = np.array(['airport','shopping_mall','metro_station','street_pedestrian','public_square','street_traffic','tram','bus','metro','park'])

f = open(prediction_results_folder + '/preds_fused.csv','w')
f.write('Id,Scene_label\n')
for i, class_id in enumerate(preds_fused):
	f.write(eval_files[i]+','+class_names[class_id]+'\n') #Give your csv text here.
f.close()

pickle.dump(preds_fused,open(prediction_results_folder + '/preds_fused.pickle','wb'))
pickle.dump(preds_soft_fused,open(prediction_results_folder + '/preds_soft_fused.pickle','wb'))


