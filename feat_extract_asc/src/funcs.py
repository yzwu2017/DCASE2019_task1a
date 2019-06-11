import os
import numpy as np
import csv
import pickle as pickle
import librosa
import yaml
import scipy
import time
import skimage





def ensure_folder_exists(folder_path):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)


def convert_to_uint8(image):
	output_image = (image - image.min())/(image.max() - image.min()) * 255
	output_image = output_image.astype(np.uint8)
	return output_image


def get_feature_filename(audio_file_path, feature_path, extension='pickle'):
	# parameters: ----------------------------
	# audio_file_path (str):	audio file's absolute path from which the features are extracted
	# feature_path (str):		the folder which the feature will be saved
	# extension (str):			file extension (Default value='pickle')
	# Returns: ----------------------------
	# feature_filename (str):	full feature file path. e.g. /home/xx/xxx/feat.pickle
	audio_filename = os.path.split(audio_file_path)[1]
	return os.path.join(feature_path, os.path.splitext(audio_filename)[0] + '.' + extension)


def extract_logmel(data, sr=48000, statistics=True, win_length=1200, hop_length=480, config = None):
	eps = np.spacing(1)
	# Windowing function
	if config['window'] == 'hamming_asymmetric':
		window = scipy.signal.hamming(win_length, sym=False)
	elif config['window'] == 'hamming_symmetric':
		window = scipy.signal.hamming(win_length, sym=True)
	elif config['window'] == 'hann_asymmetric':
		window = scipy.signal.hann(win_length, sym=False)
	elif config['window'] == 'hann_symmetric':
		window = scipy.signal.hann(win_length, sym=True)
	else:
		window = None
	# Calculate Static Coefficients
	power_spectrogram = np.abs(librosa.stft(data + eps,
					n_fft=config['n_fft'],
					win_length=win_length,
					hop_length=hop_length,
					center=True,
					window=window))**2
	mel_basis = librosa.filters.mel(sr=sr,
					n_fft=config['n_fft'],
					n_mels=config['n_mels'],
					fmin=config['fmin'],
					fmax=config['fmax'],
					)
	mel_spectrum = np.dot(mel_basis, power_spectrogram)
	# Collect the feature matrix
	feature_matrix = librosa.power_to_db(mel_spectrum)
	feature_matrix = feature_matrix.T #".T" means transpose of the matrix
	return feature_matrix


def extract_mel_spectrum(data, sr=48000, statistics=True, win_length=1200, hop_length=480, config = None):
	eps = np.spacing(1)
	# Windowing function
	if config['window'] == 'hamming_asymmetric':
		window = scipy.signal.hamming(win_length, sym=False)
	elif config['window'] == 'hamming_symmetric':
		window = scipy.signal.hamming(win_length, sym=True)
	elif config['window'] == 'hann_asymmetric':
		window = scipy.signal.hann(win_length, sym=False)
	elif config['window'] == 'hann_symmetric':
		window = scipy.signal.hann(win_length, sym=True)
	else:
		window = None
	# Calculate Static Coefficients
	power_spectrogram = np.abs(librosa.stft(data + eps,
					n_fft=config['n_fft'],
					win_length=win_length,
					hop_length=hop_length,
					center=True,
					window=window))**2
	mel_basis = librosa.filters.mel(sr=sr,
					n_fft=config['n_fft'],
					n_mels=config['n_mels'],
					fmin=config['fmin'],
					fmax=config['fmax'],
					)
	mel_spectrum = np.dot(mel_basis, power_spectrogram)
	# Collect the feature matrix
	mel_spectrum = mel_spectrum.T #".T" means transpose of the matrix
	return mel_spectrum


def delta(feat, N):
	"""Compute delta features from a feature vector sequence.
	:param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector. Dimension is like [Time, Frequency]
	:param N: For each frame, calculate delta features based on preceding and following N frames
	:returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
	"""
	if N < 1:
		raise ValueError('N must be an integer >= 1')
	NUMFRAMES = len(feat)
	denominator = 2 * sum([i**2 for i in range(1, N+1)])
	delta_feat = np.empty_like(feat)
	padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
	for t in range(NUMFRAMES):
		delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator  # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
	return delta_feat


def extract_logmel_dynamics(data, sr=48000, statistics=True, win_length=1200, hop_length=480, N_delta=4, config = None):
	eps = np.spacing(1)
	# Windowing function
	if config['window'] == 'hamming_asymmetric':
		window = scipy.signal.hamming(win_length, sym=False)
	elif config['window'] == 'hamming_symmetric':
		window = scipy.signal.hamming(win_length, sym=True)
	elif config['window'] == 'hann_asymmetric':
		window = scipy.signal.hann(win_length, sym=False)
	elif config['window'] == 'hann_symmetric':
		window = scipy.signal.hann(win_length, sym=True)
	else:
		window = None
	# Calculate Static Coefficients
	power_spectrogram = np.abs(librosa.stft(data + eps,
					n_fft=config['n_fft'],
					win_length=win_length,
					hop_length=hop_length,
					center=True,
					window=window))**2
	mel_basis = librosa.filters.mel(sr=sr,
					n_fft=config['n_fft'],
					n_mels=config['n_mels'],
					fmin=config['fmin'],
					fmax=config['fmax'],
					)
	mel_spectrum = np.dot(mel_basis, power_spectrogram)
	# Collect the feature matrix
	feature_matrix_static = librosa.power_to_db(mel_spectrum)
	feature_matrix_static = feature_matrix_static.T #".T" means transpose of the matrix
	feature_matrix_delta = delta(feat=feature_matrix_static,N=N_delta)
	feature_matrix_accel = delta(feat=feature_matrix_delta,N=N_delta)
	feature_matrix = np.array([feature_matrix_static,feature_matrix_delta,feature_matrix_accel])
	return feature_matrix




# The tic toc functions are originally copied from the following source, though they are modified a little bit.
# Ref: https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python 	
def tic():
	global startTime_for_tictoc
	startTime_for_tictoc = time.time()
	
def toc():
	if 'startTime_for_tictoc' in globals():
		print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
		del globals()['startTime_for_tictoc']
	else:
		print("Toc: start time not set")


