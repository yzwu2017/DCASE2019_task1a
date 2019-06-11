import numpy as np
import torch
import random
import cv2
import math
import torch.nn.functional as F
import random

def background_subtraction(feat_mtx):
	for sample_idx, feat in enumerate(feat_mtx):
		mean_of_feat = np.mean(feat)
		feat_mtx[sample_idx] = feat - mean_of_feat
	return feat_mtx

def stretch_tensor(tensor_in, stretch_scale_HW=(0.45,0.45), n_patches_HW=(4,4)):
	# this function can be used to do the time-frequency stretching of 4D tensor.
	image_H = tensor_in.size(2)
	image_W = tensor_in.size(3)
	n_patches_H = n_patches_HW[0]
	n_patches_W = n_patches_HW[1]
	stretch_scale_H = stretch_scale_HW[0]
	stretch_scale_W = stretch_scale_HW[1]
	# The stretched coords are derived from the original grid coordinates.
	mean_patch_H = image_H / n_patches_H
	mean_patch_W = image_W / n_patches_W
	grid_coords_H = np.linspace(1,image_H,n_patches_H+1,dtype='int')
	grid_coords_W = np.linspace(1,image_W,n_patches_W+1,dtype='int')
	# Make the stretching coordinates for H-axis
	stretch_coef_H = 2*np.random.rand(len(grid_coords_H))-1 # generate random number of range [-1,1)
	stretch_coef_H[[0,-1]] = 0
	stretch_pixels_H = mean_patch_H * stretch_scale_H * stretch_coef_H
	stretch_pixels_H = stretch_pixels_H.astype(int) 
	stretched_coords_H = grid_coords_H + stretch_pixels_H
	# Make the stretching coordinates for W-axis
	stretch_coef_W = 2*np.random.rand(len(grid_coords_W))-1 # generate random number of range [-1,1)
	stretch_coef_W[[0,-1]] = 0
	stretch_pixels_W = mean_patch_W * stretch_scale_W * stretch_coef_W
	stretch_pixels_W = stretch_pixels_W.astype(int) 
	stretched_coords_W = grid_coords_W + stretch_pixels_W
	# Creating the stretched tensor.
	tensor_out = torch.zeros(tensor_in.shape)
	for i in range(n_patches_H):
		for j in range(n_patches_W):
			patch = tensor_in[:,:,i*mean_patch_H:(i+1)*mean_patch_H,j*mean_patch_W:(j+1)*mean_patch_W]
			stretch_shape = (stretched_coords_H[i+1]-stretched_coords_H[i],stretched_coords_W[j+1]-stretched_coords_W[j])
			stretched_patch = F.upsample(patch, size=stretch_shape, mode='bilinear')
			tensor_out[:,:,stretched_coords_H[i]:stretched_coords_H[i+1],stretched_coords_W[j]:stretched_coords_W[j+1]] = stretched_patch
	# tensor_out is the stretched tensor.
	return tensor_out


# # [For Debug] Check the beta distribution
# r = np.random.beta(2.0, 2.0, size=1000)
# plt.hist(r) #histtype='stepfilled', alpha=0.2)
# plt.show()

# [For Debug]
# batch_data = torch.Tensor(np.random.random([5,1,3,2]))
# batch_labels = torch.eye(5)
def mixup_samples(batch_data, batch_labels, mix_percentage=1.0):
	# Note: In the original paper https://arxiv.org/pdf/1710.09412.pdf, the weights follow beta distribution np.random.beta(alpha, alpha,len(batch_data)), we may set alpha=0.2. 
	# Though considering the speed of sampling, uniform distribution is faster.
	mix_weights_label =  torch.Tensor(np.random.uniform(low=0.0, high=1.0, size=len(batch_data))).unsqueeze(1) 
	#mix_weights_label =  torch.Tensor(np.random.beta(0.2, 0.2, size=len(batch_data))).unsqueeze(1) 
	mix_weights_data =mix_weights_label.unsqueeze(1).unsqueeze(1)
	exotic_sample_indices = np.arange(len(batch_data))
	np.random.shuffle(exotic_sample_indices) # Shuffling the indices so that a random sample is chosen as the exotic sample.
	batch_data_mixed = (1-mix_weights_data) * batch_data + mix_weights_data * batch_data[exotic_sample_indices]
	batch_labels_mixed = (1-mix_weights_label) * batch_labels + mix_weights_label * batch_labels[exotic_sample_indices]
	if mix_percentage > 0.0 and mix_percentage < 1.0:
		n_mix = int(mix_percentage * len(batch_data))
		batch_data_mixed[n_mix:] = batch_data[n_mix:]
		batch_labels_mixed[n_mix:] = batch_labels[n_mix:]
	return batch_data_mixed, batch_labels_mixed



# [For Debug]
# import matplotlib.pyplot as plt
# batch_data = torch.Tensor(np.ones([5,1,10,10]))
# batch_data = random_erasing(batch_data)
# plt.imshow(batch_data[0,0])
# plt.show()
# reference: https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
# [Comments]: 
# the implementation is like erasing the same patch in a batch of samples
def random_erasing(batch_data, min_area=0.02, max_area=0.4, min_aspect_ratio=0.3, erase_percentage=0.5, erased_patch_value=0.0):
	for attempt in range(100): # If tried for 100 times and failed, just return the original batch data.
		area = batch_data.shape[2] * batch_data.shape[3]
		target_area = random.uniform(min_area, max_area) * area
		aspect_ratio = random.uniform(min_aspect_ratio, 1/min_aspect_ratio)
		h = int(round(math.sqrt(target_area * aspect_ratio)))
		w = int(round(math.sqrt(target_area / aspect_ratio)))	
		if h < batch_data.shape[2] and w < batch_data.shape[3]:
			x1 = random.randint(0, batch_data.shape[2] - h)
			y1 = random.randint(0, batch_data.shape[3] - w)
			batch_data[:, :, x1:x1+h, y1:y1+w] = erased_patch_value
			return batch_data
	return batch_data


