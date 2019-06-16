# 20190524 Python3 Version of Feature Extraction for ASC task.

import sys
import numpy as np
import glob #use to get file list in a folder
import soundfile as sf
import librosa #use to extract MFCC feature
import yaml #use to save and read statistics
import matplotlib.pyplot as plt
import scipy.misc
import scipy.signal
import cv2

import time
from multiprocessing import Pool
from scipy import ndimage

from src.funcs import * 

import time
import imageio

N_CORES = 4

config = {	'save_spectrograms': True, # If True, the extracted features will be saved as images for visualization purpose.
			'overwrite': False,  # Overwrite flag: Whether overwritting the existing feature file or not.
			'raw_data_folder': '/home/yzwu/DCASE2019_task1a/datasets/TAU-urban-acoustic-scenes-2019-evaluation/audio',
			'output_feature_folder': '/home/yzwu/DCASE2019_task1a/features/evaluation/logmel-128-S',
			'spectrograms_folder': '/home/yzwu/DCASE2019_task1a/features/evaluation/logmel-128-S-imgs',
			'SR': 48000,                      # The sampling frequency for feature extraction.
			'win_length_in_seconds': 0.025,   # the window length (in second). Default: 0.025
			'hop_length_in_seconds': 0.010,   # the hop length (in second). Default: 0.010
			'window': 'hamming_asymmetric',   # [hann_asymmetric, hamming_asymmetric]
			'n_fft': 2048,                    # FFT length       
			'n_mels': 128,                     # Number of MEL bands used
			'fmin': 20,                       # Minimum frequency when constructing MEL bands
			'fmax': 24000,                    # Maximum frequency when constructing MEL band
			'medfil_kernel': [11,1],          # Kernel size for Medium Filtering.
			'medfil_kernel2': [51,1], 		  # Kernel size for Medium Filtering II.
			}

# config_file = 'feature_config.yaml'
# with open(config_file,'r') as f:
# 	config = yaml.load(f)

save_spectrograms = config['save_spectrograms'] # If True, the extracted features will be saved as images for visualization purpose.
overwrite = config['overwrite'] # Overwrite flag: Whether overwritting the existing feature file or not.


raw_data_folder = config['raw_data_folder']
output_feature_folder = config['output_feature_folder']
spectrograms_folder = config['spectrograms_folder']
feature_type = 'logmel'

#=======================
# Setting for feature extractions
#=======================

win_length = int(config['win_length_in_seconds'] * config['SR'])
hop_length = int(config['hop_length_in_seconds'] * config['SR'])


raw_audio_list = glob.glob(raw_data_folder + '/*.wav')
n_audio = len(raw_audio_list)

# split the whole audio list into sub-lists.
n_audio_split = int(np.ceil(n_audio / float(N_CORES)))
sub_lists = []
n_remains = n_audio
current_idx = 0
for i in range(N_CORES):
	if n_audio >= n_audio_split:
		sublist = raw_audio_list[current_idx : current_idx+n_audio_split]
		sub_lists.append(sublist)
		n_remains = n_remains - n_audio_split
		current_idx = current_idx+n_audio_split
	else:
		sublist = raw_audio_list[current_idx:]
		sub_lists.append(sublist)

ensure_folder_exists(output_feature_folder)
if save_spectrograms:
	ensure_folder_exists(spectrograms_folder)


def extract_feature_batch(audio_list):
	count = 0
	for file_id, audio_file_path in enumerate(audio_list):
		start_time = time.time()
		current_feature_file = get_feature_filename(audio_file_path, output_feature_folder, extension=feature_type)
		if not os.path.isfile(current_feature_file) or overwrite:
			# Load audio data
			if os.path.isfile(audio_file_path):
				data, samplerate = sf.read(audio_file_path)
			else:
				raise IOError("Audio file not found [%s]" % os.path.split(audio_file_path)[1])
			#=================================
			# Extract features
			#=================================
			if feature_type == 'logmel':
				data_left = data[:,0]
				data_right = data[:,1]
				logmel_left = extract_logmel(data=data_left, sr=config['SR'], win_length=win_length, hop_length=hop_length, config=config)
				logmel_right = extract_logmel(data=data_right, sr=config['SR'], win_length=win_length, hop_length=hop_length, config=config)
				logmel_mid = (logmel_left + logmel_right) / 2.0	

				# Medium Filtering to extract Low-Medium-High temporal variations Logmel features.
				logmel_notHigh = scipy.signal.medfilt(logmel_mid, kernel_size=config['medfil_kernel'])
				logmel_high = logmel_mid - logmel_notHigh		
				logmel_low = scipy.signal.medfilt(logmel_notHigh, kernel_size=config['medfil_kernel2'])
				logmel_medium = logmel_notHigh - logmel_low
				logmel_lmh = np.array([logmel_low,logmel_medium,logmel_high])

				feature_data = {#'feat_left': logmel_left,
								#'feat_right': logmel_right,
								# 'feat_medium': logmel_medium,
								'feat_lmh': logmel_lmh,
								}
				# Save feature data
				pickle.dump(feature_data, open(current_feature_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)		
				if save_spectrograms:
					img_file_name = current_feature_file.split('/')[-1]
					img_file_name = img_file_name.split('.')[0]

					# specgram_img = logmel_mid.T
					# specgram_img = np.flip(specgram_img,axis=0) #Making the bottom of the image be the low-frequency part.
					# specgram_img = (specgram_img - np.min(specgram_img)) / (np.max(specgram_img) - np.min(specgram_img))
					# specgram_img = skimage.img_as_ubyte(specgram_img)
					# imageio.imwrite(spectrograms_folder + '/' + img_file_name +'.jpg', specgram_img)

					specgram_img = logmel_medium.T
					specgram_img = np.flip(specgram_img,axis=0) #Making the bottom of the image be the low-frequency part.
					specgram_img = 255 * (specgram_img - np.min(specgram_img)) / (np.max(specgram_img) - np.min(specgram_img))
					specgram_img = specgram_img.astype(np.uint8)
					imageio.imwrite(spectrograms_folder + '/' + img_file_name +'-m.jpg', specgram_img)

					specgram_img = logmel_low.T
					specgram_img = np.flip(specgram_img,axis=0) #Making the bottom of the image be the low-frequency part.
					specgram_img = 255 * (specgram_img - np.min(specgram_img)) / (np.max(specgram_img) - np.min(specgram_img))
					specgram_img = specgram_img.astype(np.uint8)
					imageio.imwrite(spectrograms_folder + '/' + img_file_name +'-l.jpg', specgram_img)


					specgram_img = logmel_high.T
					specgram_img = np.flip(specgram_img,axis=0) #Making the bottom of the image be the low-frequency part.
					specgram_img = 255 * (specgram_img - np.min(specgram_img)) / (np.max(specgram_img) - np.min(specgram_img))
					specgram_img = specgram_img.astype(np.uint8)
					imageio.imwrite(spectrograms_folder + '/' + img_file_name +'-h.jpg', specgram_img)

		count = count + 1
		elapsed = time.time() - start_time
		print("[Time: %.2fs] Progress %.1f%% | " % (elapsed,(file_id+1) / float(len(audio_list)) * 100) +  os.path.split(audio_file_path)[1] + "                              ", end='\r')


# Save feature configuration
with open(output_feature_folder + '/feature.config','w') as yaml_file:
	yaml.dump(config, yaml_file, default_flow_style=False)	


#========================
# Start Feature Extraction Using Multiple Cores.
#========================
#mark the start time
startTime = time.time()
#create a process Pool with N_CORES processes
pool = Pool(processes=N_CORES)
# map doWork to availble Pool processes
pool.map(extract_feature_batch, sub_lists)
#mark the end time
endTime = time.time()
#calculate the total time it took to complete the work
workTime =  endTime - startTime
#print results
print("The job took " + str(workTime) + " seconds to complete")
