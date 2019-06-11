# 20190523 For DCASE2019 task(1)-subtask(A)
# The script is used to train and test the AlexNet-S model with logMel-128-S feature.
# The model dict will be saved after training, which can be further used to generate prediction file for leaderboard and evaluation data.

import numpy as np
import pickle
import glob
import random
import yaml
import copy
from sklearn.utils import shuffle
import itertools
from src.functions_asc import *
from src.feature_preprocess_20190522 import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def computeLoss_GPU(x, y, batch_size, loss_func, model):
	# Compute the loss given the model, loss and a set of data, label.
	# inputs:
	#		x:				the data, torch.Tensor with dimension [n_samples, input_that_match_the_model]
	#		y:				the label, torch.Tensor with dimension [n_samples, output_that_match_the_model]
	#		loss_func:	the loss function you have defined.
	#		model:		the model you have trained.
	#		(Notice: The torch.Tensor and is passed by reference. )
	# output:
	#		total_loss:			loss of the data given the model.
	model.eval() # Set model to evaluation mode (disable dropout layer)
	step = 0
	total_loss = 0
	n_samples = y.shape[0]
	batch_indices = torch.Tensor(np.arange(batch_size)).long()
	while batch_indices[0] < n_samples:
		# Get a batch of training data and label
		if batch_indices[0] +  batch_size <= n_samples:
			# If the indices are all valid, we directly use them.
			batch_x = x[batch_indices].cuda(gpu)
			batch_y = y[batch_indices].cuda(gpu)
			# Update indices
			batch_indices = batch_indices + batch_size
		else:
			# If the indices are out of range, we need to discard some of them.
			num_remain_audio = n_samples - batch_indices[0]
			batch_indices = batch_indices[0 : num_remain_audio]
			# Use the valid indices to get the batch data and label.
			batch_x = x[batch_indices]
			batch_y = y[batch_indices] 
			# Update indices
			batch_indices = batch_indices + num_remain_audio
		# Make the batch data dimension match the input spec. of nn.Conv2D
		#batch_x = batch_x.unsqueeze(1)
		batch_x = Variable(batch_x).cuda(gpu)
		batch_y = Variable(batch_y).cuda(gpu)
		# Get model output and compute the loss
		output = model(batch_x)
		batch_loss = loss_func(output, batch_y)
		total_loss = total_loss + float(batch_loss.cpu().data.numpy())
		step = step + 1
	# Average the total_loss
	total_loss = total_loss / step
	return total_loss


#Load the configurations
with open('general_config.yaml','r') as f:
	config = yaml.load(f,Loader=yaml.FullLoader)

doBgSub = False
results_folder_name = 'results-logmel128S-AlexNetS-Mixup-20eps'
gpu = 0 # the gpu used for running the script.
def getModel(gpu):
	return AlexNetS().cuda(gpu)

def data_augmentation(batch_x,batch_y):
	#batch_x = stretch_tensor(batch_x, stretch_scale_HW=(0.2,0), n_patches_HW=(4,1))
	batch_x, batch_y = mixup_samples(batch_x, batch_y, mix_percentage=1.0)
	#batch_x = random_erasing(batch_x, min_area=0.02, max_area=0.4, min_aspect_ratio=0.3, erase_percentage=0.5, erased_patch_value=0.0)
	return batch_x, batch_y


#==============================================
#Load Feature Data 
#==============================================
print('Loading Data and Feature Pre-processing. ')


## Define the classes corresponding label index
num_of_classes = config['class_spec']['asc2019']['number_of_classes']
label_dict=copy.copy(config['class_spec']['asc2019']['index_assignment'])

development_data_path = '/home/yzwu/DCASE2019_task1a/features/development/logmel-128-HF11LF51'
train_label_doc = '/home/yzwu/DCASE2019_task1a/datasets/TAU-urban-acoustic-scenes-2019-development/evaluation_setup_compatibleFormat/fold1_train.csv' # each row is like [filename,label,location,device_id]
test_label_doc = '/home/yzwu/DCASE2019_task1a/datasets/TAU-urban-acoustic-scenes-2019-development/evaluation_setup_compatibleFormat/fold1_evaluate.csv' # each row is like [filename,label,location,device_id]
file_extension = 'logmel'
FEATURE_DIMENSION = 128



def load_data_3c(data_path, label_doc, file_extension):
	## Read the label document for the data
	with open(label_doc, "r") as text_file:
		lines = text_file.read().split('\n')
		# post-processing the document.
		for idx, ele in enumerate(lines):
			lines[idx]=lines[idx].split('\t')
			lines[idx][0]=lines[idx][0].split('/')[-1].split('.')[0]
	lines = lines[1:] # delete the first row which is the table head.
	lines  = [ele for ele in lines if ele != ['']] # only keep the non-empty elements
	for idx, ele in enumerate(lines):
		lines[idx][-1]=lines[idx][-1].split('\r')[0]
	label_info=np.array(lines)
	del lines[:]
	## Read the feature matrix for each audio file (based on the label document)
	feat_mtx=[]
	for [filename,label,location,device_id] in label_info:
		filepath = data_path + '/' + filename + '.' + file_extension
		with open(filepath,'rb') as f:
			temp=pickle.load(f, encoding='latin1') # encoding='latin1' for loading python2 pickle file in python3.
			feat_mtx.append(temp['feat_lmh'])
	#feat_mtx=np.array(feat_mtx)
	## Zero Padding (in time-frequency representation, we actually pad the audio sequences with vectors with low values of dB(representing silence), to make the sequence length a multiple of pad_unit. )
	pad_unit = 128
	low_energy_value = -80 # -80dB to indicate ``silence'' 
	for idx, feat in enumerate(feat_mtx):
		feat_len = feat.shape[-2]
		if feat_len % pad_unit != 0:
			n_pad = pad_unit - feat_len % pad_unit
			pad_mtx = low_energy_value * np.ones([3,n_pad,FEATURE_DIMENSION])
			feat_padded = np.concatenate((feat, pad_mtx),axis = -2)
			feat_mtx[idx] = feat_padded
	feat_mtx = np.array(feat_mtx)
	return feat_mtx, label_info


feat_train, label_train = load_data_3c(development_data_path, train_label_doc, file_extension)
feat_test, label_test = load_data_3c(development_data_path, test_label_doc, file_extension)

# Background Subtraction
if doBgSub == True:
	feat_train = background_subtraction(feat_train)
	feat_test = background_subtraction(feat_test)


#==============================================
#Model Structure
#==============================================
# Hyper Parameters
hyper_params = 	{			'NUM_OF_CLASSES': num_of_classes, 
							'EPOCH': 20, 							# number of training epoches
							'BATCH_SIZE': 100,						# batch size (the number represent number of audio files) (default is 60)
							'INPUT_SIZE': FEATURE_DIMENSION,		# rnn input size
							'LR': 2e-4,								# learning rate (default is 0.00001)
							'WEIGHT_DECAY_COEFFICIENT': 0.0015
							}
							
EPOCH = hyper_params['EPOCH']
BATCH_SIZE = hyper_params['BATCH_SIZE']
INPUT_SIZE = hyper_params['INPUT_SIZE']						
LR = hyper_params['LR']			
WEIGHT_DECAY_COEFFICIENT = hyper_params['WEIGHT_DECAY_COEFFICIENT']


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


print('Initialize Model Structure. ')
myModel = getModel(gpu)
#params = list(seg_rnn.parameters()) + list(audio_rnn.parameters()) # if you have two modules to train, concatenate their parameters lists.
params = myModel.parameters()

optimizer = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY_COEFFICIENT)	#optimizer all rnn parameters
loss_func = nn.BCEWithLogitsLoss().cuda(gpu) 

result_folder = results_folder_name
ensure_folder_exists(result_folder)
#==============================================
#Feature Segmentation
#==============================================
# In this code block, we will cut the feature of each audio file into segments (time slices).

hop_size = 128			# Default is 50
segment_length = 128	# Default is 100
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

feat_train_seg, label_train_seg = feature_segmentation_3c(feat_train, label_train, hop_size, segment_length)
feat_test_seg, label_test_seg = feature_segmentation_3c(feat_test, label_test, hop_size, segment_length)


def process_label_info_as_onehot_asc2019(label_info):
		# Generate a list of digits that represent the class for each sample.
		label_info_classidx = []
		for idx, [filename, classname, location, device_id] in enumerate(label_info):
			class_idx = label_dict[classname]
			label_info_classidx.append(np.eye(num_of_classes)[class_idx])
		label_info_classidx = np.array(label_info_classidx)
		return label_info_classidx


#==============================================
#Feature Normalization
#==============================================
# In this code block, we will first normalize the data to zero mean and unit variance

train_x = feat_train_seg 
train_y = process_label_info_as_onehot_asc2019(label_train_seg)
test_x = feat_test_seg
test_y = process_label_info_as_onehot_asc2019(label_test_seg)

# Compute the mean and std of the training set.
mean_train = np.mean(feat_train_seg.reshape([-1,FEATURE_DIMENSION]),axis=0)
std_train = np.std(feat_train_seg.reshape([-1,FEATURE_DIMENSION]),axis=0)
std_train = numeric_stable_std(std_train) #ensure there is no 0, NaN or infinity in the std vector.

# Do feature normalization (zero mean and unit variance)
train_x = (train_x - mean_train) / std_train
test_x = (test_x - mean_train) / std_train

# Save the mean and variance for future testing use.
ensure_folder_exists(result_folder)
normalization_stats = {'mean_train': mean_train, 'std_train': std_train}
pickle.dump(normalization_stats, open(result_folder + '/normstats.pickle','wb'))	

#==============================================
#Model Training
#==============================================
print('Start Model Training. ')


train_x = torch.Tensor(train_x)
train_y = torch.Tensor(train_y)

test_x = torch.Tensor(test_x)
test_y = torch.Tensor(test_y)

# Initialize lists for saving the loss information
train_loss = []
test_loss = []

step = 0 # For Training Progress Display
for epoch in range(EPOCH):
	# Reshape and then do shuffling according to the first dimension.
	rand_perm = torch.randperm(train_y.shape[0]).long()
	train_x = train_x[rand_perm]
	train_y = train_y[rand_perm]
	
	# Learning Rate Decay
	if epoch % 4 == 0 and epoch != 0:
		LR = LR / 2.0
		optimizer = torch.optim.Adam(myModel.parameters(), lr=LR, weight_decay=WEIGHT_DECAY_COEFFICIENT)

	# Training (1 epoch)
	n_train = train_y.shape[0]
	batch_indices = torch.Tensor(np.arange(BATCH_SIZE)).long()
	while batch_indices[0] < n_train:
		# Get a batch of training data and label
		if batch_indices[0] +  BATCH_SIZE <= n_train:
			# If the indices are all valid, we directly use them.
			batch_x = train_x[batch_indices]
			batch_y = train_y[batch_indices]
			batch_indices = batch_indices + BATCH_SIZE
		else:
			# If the indices are out of range, we need to discard some of them.
			num_remain_audio = n_train - batch_indices[0]
			batch_indices = batch_indices[0 : num_remain_audio]
			# Use the valid indices to get the batch data and label.
			batch_x = train_x[batch_indices]
			batch_y = train_y[batch_indices]
			batch_indices = batch_indices + num_remain_audio
		
		#batch_x = batch_x.unsqueeze(1)

		batch_x, batch_y = data_augmentation(batch_x,batch_y)
		
		batch_x = Variable(batch_x).cuda(gpu)
		batch_y = Variable(batch_y).cuda(gpu)
		
		# Train with the batch data and label
		myModel.train()
		output = myModel(batch_x)
		loss = loss_func(output, batch_y) 
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		step = step + 1
		
		# print 'One step finished! '
		
		# Display the Loss for each "display_steps" steps (default is 50)
		display_steps = 50
		if step % display_steps == 0:
			train_loss.append(float(loss.cpu().data.numpy()))
			loss_validationSet = computeLoss_GPU(	x = test_x,
													y = test_y,
													batch_size = BATCH_SIZE,
													loss_func = loss_func, 
													model = myModel)
			test_loss.append(loss_validationSet)
			print('Epoch:%d' % epoch, '| TrainLoss: %.3f' % train_loss[-1], '| TestLoss: %.3f' % test_loss[-1])





# Save the trained model
ensure_folder_exists(os.getcwd() + '/' + result_folder)
torch.save(myModel.state_dict(), result_folder + '/myModel.dict')

# After training, plot the learning curve.
plotLearningCurve(train_loss, test_loss, n_steps = step, result_folder_name=result_folder)

training_curve_stats = {'train_loss': train_loss, 'test_loss': test_loss}
pickle.dump(training_curve_stats, open(result_folder + '/training_curve_stats.pickle','wb'))	

#==============================================
# Evaluate the Model Performance on the Test Fold
#==============================================	

# Feature Normalization
eval_x = feat_test_seg
eval_y_seg = process_label_info_as_onehot_asc2019(label_test_seg)
eval_x = (eval_x - mean_train) / std_train

# 4. Model Evaluation
print('Evaluating the Model. ')

eval_x = torch.Tensor(eval_x)
#eval_y = torch.Tensor(eval_y)

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
		#batch_y = eval_y[batch_indices].cuda(gpu)
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
		#batch_y = eval_y[batch_indices] 
		# Specify the indices of model outputs for this batch
		eval_outputs_indices = batch_indices.cpu().numpy()
		# Update indices
		batch_indices = batch_indices + num_remain_audio
		
	# Make the batch data dimension match the input spec. of nn.Conv2D
	#batch_x = batch_x.unsqueeze(1)
	
	batch_x = Variable(batch_x).cuda(gpu)
	#batch_y = Variable(batch_y).cuda(gpu)
	
	# Get the model output given the test batch
	eval_out = myModel(batch_x)
	eval_out = torch.softmax(eval_out,-1) # This is the model output
	eval_outputs_seg[eval_outputs_indices] = eval_out.cpu().data.numpy()



## Obtain the sample-level output vector from segment-level output vectors.
def getSampleResultFromSegments(label_seg):
	eval_outputs = [] # the model outputs.
	eval_y = [] # the ground-truth label for the data.
	current_file = None
	for idx, [fname, cname, location, device_id] in enumerate(label_seg):
		if current_file == None: # If we are dealing with the first segment.
			current_file = fname
			current_label = eval_y_seg[idx]
			output_avg_seg = eval_outputs_seg[idx]
			count_of_seg = 1
		elif current_file == fname: # If this segment belongs to the same audio sample as the previous segment.
			output_avg_seg = output_avg_seg + eval_outputs_seg[idx]
			count_of_seg = count_of_seg + 1
		elif current_file != fname: # If this segment belongs to a new audio sample
			# We have obtain all segments for the last audio sample,so save it.
			output_avg_seg = output_avg_seg / float(count_of_seg)
			eval_outputs.append(output_avg_seg)
			eval_y.append(current_label)
			# Update the current file to the new file.
			current_file = fname
			current_label = eval_y_seg[idx]
			output_avg_seg = eval_outputs_seg[idx]
			count_of_seg = 1

	# save the averaged output vector for the last audio sample.
	output_avg_seg = output_avg_seg / float(count_of_seg)
	eval_outputs.append(output_avg_seg)
	eval_y.append(current_label)

	# Convert list to numpy array.
	eval_outputs = np.array(eval_outputs)
	eval_y = np.array(eval_y)

	return eval_outputs, eval_y

eval_outputs, eval_y = getSampleResultFromSegments(label_test_seg)

#==============================================
# Result Analysis
#==============================================	

## Obtain predictions
# Convert outputs from one-hot vector to labels.
predictions = 99 * np.ones(len(eval_outputs))
for i in range(len(eval_outputs)):
	predictions[i] = np.argmax(eval_outputs[i])
predictions = predictions.astype('int')

# Do the same for segment-level result.
predictions_seg = 99 * np.ones(len(eval_outputs_seg))
for i in range(len(eval_outputs_seg)):
	predictions_seg[i] = np.argmax(eval_outputs_seg[i])
predictions_seg = predictions_seg.astype('int')
	
## Print the overall accuracy 
sample_truth = [np.argmax(pred) for pred in eval_y]
accuracy = sum(predictions == sample_truth) / float(len(predictions))
print('Model Evaluation: | accuracy: %.3f' % accuracy)

seg_truth = [np.argmax(pred) for pred in eval_y_seg]
accuracy_seg = sum(predictions_seg == seg_truth) / float(len(predictions_seg))
print('Model Evaluation: | segment-level accuracy: %.3f' % accuracy_seg)



## Calculate the confusion matrices
confusion_mtx = getConfusionMatrix(pred_labels=predictions, true_labels=sample_truth, n_classes = num_of_classes)
confusion_mtx_seg = getConfusionMatrix(pred_labels=predictions_seg, true_labels=seg_truth, n_classes = num_of_classes)
class_names_short = np.array(['air','shop','metro_s','str_p','p_squ','str_t','tram','bus','metro','park'])
class_names = np.array(['airport','shopping_mall','metro_station','street_pedestrian','public_square','street_traffic','tram','bus','metro','park'])

# Calculate accuracy for each class, and save the results.
ensure_folder_exists(os.getcwd() + '/' +result_folder)
result_file = open(result_folder + '/result_analysis.txt', 'w')
print('{:25}'.format('Sample-level accuracy') + '{:^.3f}'.format(accuracy), file=result_file)
print('{:25}'.format('Segment-level accuracy') + '{:^.3f}'.format(accuracy_seg), file=result_file)

print('-------------------', file=result_file)
print('Confusion Matrix (Sample): ', file=result_file)
print(confusion_mtx, file=result_file)

print('-------------------', file=result_file)
print('Confusion Matrix (Segment): ', file=result_file)
print(confusion_mtx_seg, file=result_file)

print('-------------------', file=result_file)
print('The change of loss during training: ', file=result_file)
for idx in range(len(train_loss)):
	print('{:15}'.format('Train Loss: ') + '{:^.3f}'.format(train_loss[idx]) + ' | ' + '{:15}'.format('Val. Loss: ') + '{:^.3f}'.format(test_loss[idx]), file=result_file)

result_file.close()
	
# Plot and save confusion matrix
plt.clf()
plt.figure(figsize = (10,10))
plot_confusion_matrix(confusion_mtx, classes=class_names_short, normalize=False, savefig_name = 'cnf_mtx.png', save_folder_name = result_folder, 
						title='Confusion matrix (Audio Samples)')
					  

plt.clf()
plt.figure(figsize = (10,10))
plot_confusion_matrix(confusion_mtx_seg, classes=class_names_short, normalize=False, savefig_name = 'cnf_mtx_seg.png', save_folder_name = result_folder, 
						title='Confusion matrix (Audio Segments)')



# ## Generate text transcription for audio files
# predictions_seg_info = []
# for idx, [fname, cname, location, device_id] in enumerate(label_test_seg):
# 	predicted_cname = class_names[predictions_seg[idx]]
# 	predictions_seg_info.append([fname, predicted_cname])
# predictions_seg_info = np.array(predictions_seg_info)

# current_file = None
# frame_duration = 0.01
# for idx, [fname, predicted_cname] in enumerate(predictions_seg_info):
# 	if current_file == None: # If we are dealing with the first segment.
# 		current_time = 0.00
# 		current_file = fname
# 		ensure_folder_exists(os.getcwd() + '/' + result_folder + '/transcription')
# 		trans_file = open(result_folder + '/transcription' + '/' + current_file + '.txt', 'w')
# 		# Write one row trans.
# 		print('{:^.3f}'.format(current_time) + '\t' + '{:^.3f}'.format(current_time + hop_size * frame_duration) + '\t' + predicted_cname, file=trans_file)
# 		current_time = current_time + hop_size * frame_duration
# 	elif current_file == fname: # If this segment belongs to the same audio sample as the previous segment.
# 		# Write one row trans.
# 		print('{:^.3f}'.format(current_time) + '\t' + '{:^.3f}'.format(current_time + hop_size * frame_duration) + '\t' + predicted_cname, file=trans_file)
# 		current_time = current_time + hop_size * frame_duration
# 	elif current_file != fname: # If this segment belongs to a new audio sample
# 		trans_file.close()
# 		current_time = 0.00
# 		current_file = fname
# 		ensure_folder_exists(os.getcwd() + '/' + result_folder + '/transcription')
# 		trans_file = open(result_folder + '/transcription' + '/' + current_file + '.txt', 'w')
# 		# Write one row trans.
# 		print(trans_file, '{:^.3f}'.format(current_time) + '\t' + '{:^.3f}'.format(current_time + hop_size * frame_duration) + '\t' + predicted_cname, file=trans_file)
# 		current_time = current_time + hop_size * frame_duration

# trans_file.close()



# ## Calculate the Degree of Confusion
# # The output file format is like:
# # -----
# # filename1	true_class	n_segments	degree_of_cofusion	is_final_prediction_correct
# # filename2	true_class	n_segments	degree_of_cofusion	is_final_prediction_correct
# # ...
# # filenameN	true_class	n_segments	degree_of_cofusion	is_final_prediction_correct
# # -----
# # The degree of confusion (DOC) is defined as the porportion of the mis-classified segments in a audio file.
# # For example, an audio file is divided into 10 segments, and 6 of the segments are classified correctly, 4 of them are mis-classified into other classes, then DOC = 0.4.
# # The boolean variable "is_final_prediction_correct" indicates whether the audio sample is correctly classified.
# def genTranscriptionAnalysis(predictions, eval_y, predictions_seg_info, label_test_seg):
# 	ensure_folder_exists(os.getcwd() + '/' + result_folder)
# 	trans_file = open(result_folder + '/transcription_analysis.txt', 'w')

# 	# An array indicates if the samples are correctly classified.
# 	sample_correctness = (predictions == eval_y)
# 	sample_idx = 0

# 	current_file = None
# 	for idx, [fname, predicted_cname] in enumerate(predictions_seg_info):
# 		if current_file == None: # If we are dealing with the first segment.
# 			n_correct, n_wrong = 0, 0
# 			current_file = fname
# 			current_file_true_label = label_test_seg[idx][1]
# 			if predicted_cname == current_file_true_label:
# 				n_correct = n_correct + 1
# 			else:
# 				n_wrong = n_wrong + 1
# 		elif current_file == fname: # If this segment belongs to the same audio sample as the previous segment.
# 			if predicted_cname == current_file_true_label:
# 				n_correct = n_correct + 1
# 			else:
# 				n_wrong = n_wrong + 1
# 		elif current_file != fname: # If this segment belongs to a new audio sample
# 			# save the previous segment's information.
# 			n_total = int(n_correct + n_wrong)
# 			print(current_file + '\t' + current_file_true_label + '\t' + '{:^d}'.format(n_total) + '\t' + '{:^.3f}'.format(n_wrong / float(n_total)) + '\t' + '{:^d}'.format(sample_correctness[sample_idx]), file=trans_file)
# 			# Because a new audio sample is encountered, update the sample_idx
# 			sample_idx = sample_idx + 1
# 			# reset the current file and its information
# 			n_correct, n_wrong = 0, 0
# 			current_file = fname
# 			current_file_true_label = label_test_seg[idx][1]
# 			if predicted_cname == current_file_true_label:
# 				n_correct = n_correct + 1
# 			else:
# 				n_wrong = n_wrong + 1


# 	# write a row with the information of the last audio sample.
# 	n_total = int(n_correct + n_wrong)
# 	print(current_file + '\t' + current_file_true_label + '\t' + '{:^d}'.format(n_total) + '\t' + '{:^.3f}'.format(n_wrong / float(n_total)) + '\t' + '{:^d}'.format(sample_correctness[sample_idx]), file=trans_file)

# 	trans_file.close()
# 	return

# genTranscriptionAnalysis(predictions, sample_truth, predictions_seg_info, label_test_seg)
