"""
Tools for the prediction of spike trains from calcium traces.
"""

from scipy.signal import resample
from numpy import percentile, asarray, arange, zeros, where, repeat, sort, cov, mean, std, ceil
from numpy import vstack, hstack, argmin
from numpy.random import rand
from copy import copy, deepcopy
from cmt.models import MCGSM, STM, Poisson
from cmt.nonlinear import ExponentialFunction, BlobNonlinearity
from cmt.tools import generate_data_from_image, extract_windows
from cmt.transforms import PCATransform
from cmt.utils import random_select
from numpy import any, isinf

def preprocess(data, fps=100., verbosity=0):
	"""
	Normalize calcium traces and spike trains.

	This function does three things:
		1. Remove any linear trends using robust linear regression.
		2. Normalize the range of the calcium trace by the 5th and 80th percentile.
		3. Change the sampling rate of the calcium trace and spike train.

	@type  data: list
	@param data: list of dictionaries containing calcium/fluorescence traces

	@type  fps: float
	@param fps: desired sampling rate of signals

	@type  verbosity: int
	@param verbosity: output progress if positive

	@rtype: list
	@return: list of preprocessed recordings
	"""

	data = deepcopy(data)

	for k in range(len(data)):
		if verbosity > 0:
			print 'Preprocessing calcium trace {0}...'.format(k)

		# remove any linear trends
		x = arange(data[k]['calcium'].size)
		a, b = robust_linear_regression(x, data[k]['calcium'])

		data[k]['calcium'] = data[k]['calcium'] - (a * x + b)

		# normalize dispersion
		calcium05 = percentile(data[k]['calcium'],  5)
		calcium80 = percentile(data[k]['calcium'], 80)

		if calcium80 - calcium05 > 0.:
			data[k]['calcium'] = (data[k]['calcium'] - calcium05) / (calcium80 - calcium05)

		# compute spike times if binned spikes are given
		if 'spikes' in data[k] and 'spike_times' not in data[k]:
			spikes = asarray(data[k]['spikes'].ravel(), dtype='uint16')

			# compute spike times in milliseconds
			spike_times = where(spikes > 0)[0]
			spike_times = repeat(spike_times, spikes[spike_times])
			spike_times = (spike_times + rand(*spike_times.shape)) * (1000. / data[k]['fps'])

			data[k]['spike_times'] = sort(spike_times).reshape(1, -1)

		# number of samples after update of sampling rate
		num_samples = int(float(data[k]['calcium'].size) * fps / data[k]['fps'] + .5)

		if num_samples != data[k]['calcium'].size:
			# factor by which number of samples will be changed
			factor = num_samples / float(data[k]['calcium'].size)

			# resample calcium signal
			data[k]['calcium'] = resample(data[k]['calcium'].ravel(), num_samples).reshape(1, -1)
			data[k]['fps'] = data[k]['fps'] * factor

		if 'spike_times' in data[k] and ('spikes' not in data[k] or num_samples != data[k]['spikes'].size):
			# spike times in bins
			spike_times = asarray(data[k]['spike_times'] * (data[k]['fps'] / 1000.), dtype=int).ravel()

			# create binned spike train
			data[k]['spikes'] = zeros([1, num_samples], dtype='uint16')
			for t in spike_times:
				data[k]['spikes'][0, t] += 1

	return data



def train(data,
		num_valid=0,
		num_models=1,
		var_explained=95.,
		window_length=1000.,
		finetune=False,
		keep_all=True,
		verbosity=1,
		model_parameters={},
		training_parameters={}):
	"""
	Trains STMs on the task of predicting spike trains from calcium traces.

	@type  data: list
	@param data: list of dictionaries containig calcium/fluorescence traces
	
	@type  num_models: int
	@param num_models: to counter local optima and other randomness, multiple models can be trained

	@type  var_explained: float
	@param var_explained: controls the number of principal components used to represent calcium window

	@type  window_length: int
	@param window_length: size of calcium window used as input to STM (in milliseconds)

	@type  finetune: bool
	@param finetune: if True, replace nonlinearity with BlobNonlinearity in second optimization step

	@type  keep_all: bool
	@param keep_all: if False, only keep the best of all trained models

	@rtype: dict
	@return: dictionary containing trained models and things needed for preprocessing
	"""

	model_parameters.setdefault('num_components', 2)
	model_parameters.setdefault('num_features', 2)
	model_parameters.setdefault('nonlinearity', ExponentialFunction)
	model_parameters.setdefault('distribution', Poisson)

	training_parameters.setdefault('max_iter', 1000)
	training_parameters.setdefault('val_iter', 1)
	training_parameters.setdefault('val_look_ahead', 100)
	training_parameters.setdefault('threshold', 1e-9)

	# turn milliseconds into bins
	window_length = int(ceil(window_length / 1000. * data[0]['fps']) + .5) # bins

	input_mask = zeros([2, window_length], dtype='bool')
	input_mask[0] = True

	output_mask = zeros([2, window_length], dtype='bool')
	output_mask[1, window_length / 2] = True

	if verbosity > 0:
		print 'Extracting inputs and outputs...'

	for entry in data:
		# extract windows from fluorescence trace and corresponding spike counts
		entry['inputs'], entry['outputs'] = generate_data_from_image(
			vstack([entry['calcium'], entry['spikes']]), input_mask, output_mask)

	inputs = hstack(entry['inputs'] for entry in data)

	if verbosity > 0:
		print 'Performing PCA...'

	pca = PCATransform(inputs, var_explained=var_explained)

	if verbosity > 0:
		print 'Reducing dimensionality of data...'

	for entry in data:
		entry['inputs'] = pca(entry['inputs'])

	models = []

	for _ in range(num_models):
		if verbosity > 0:
			print 'Training STM...'

		model = STM(
			dim_in_nonlinear=pca.pre_in.shape[0],
			dim_in_linear=0,
			**model_parameters)

		if num_valid > 0:
			idx = random_select(num_valid, len(data))

			inputs_train = hstack(entry['inputs'] for k, entry in enumerate(data) if k in idx)
			inputs_valid = hstack(entry['inputs'] for k, entry in enumerate(data) if k not in idx)
			outputs_train = hstack(entry['outputs'] for k, entry in enumerate(data) if k in idx)
			outputs_valid = hstack(entry['outputs'] for k, entry in enumerate(data) if k not in idx)

			inputs_outputs = (inputs_train, outputs_train, inputs_valid, outputs_valid)
		else:
			inputs = hstack(entry['inputs'] for entry in data)
			outputs = hstack(entry['outputs'] for entry in data)

			inputs_outputs = (inputs, outputs)

		# train model
		model.train(*inputs_outputs, parameters=training_parameters)

		if finetune:
			if verbosity > 0:
				print 'Finetuning STM...'

			# use flexible nonlinearity
			model.nonlinearity = BlobNonlinearity(num_components=3)

			# train only nonlinearity
			training_parameters_copy = copy(training_parameters)
			training_parameters_copy['train_predictors'] = False
			training_parameters_copy['train_biases'] = False
			training_parameters_copy['train_features'] = False
			training_parameters_copy['train_weights'] = False
			training_parameters_copy['train_nonlinearity'] = True
			model.train(*inputs_outputs, parameters=training_parameters_copy)

			# train all parameters jointly
			training_parameters_copy = copy(training_parameters)
			training_parameters_copy['train_nonlinearity'] = True
			model.train(*inputs_outputs, parameters=training_parameters_copy)

		models.append(model)

	if not keep_all:
		if verbosity > 0:
			print 'Only keep STM with best performance...'

		inputs = hstack(entry['inputs'] for entry in data)
		outputs = hstack(entry['outputs'] for entry in data)

		# compute negativ log-likelihoods
		logloss = []
		for model in models:
			logloss.append(model.evaluate(inputs, outputs))

		# pick best model
		models = [models[argmin(logloss)]]

	return {
		'pca': pca,
		'input_mask': input_mask,
		'output_mask': output_mask,
		'models': models}



def predict(data, results, verbosity=1):
	"""
	Predicts spike trains from calcium traces using spiking neuron models.
	"""

	if type(data) is dict:
		data = [data]
	if type(data) is not list or (len(data) > 0 and type(data[0]) is not dict):
		data = [{'calcium': data}]

	for entry in data:
		# extract windows from fluorescence trace and reduce dimensionality
		entry['inputs'] = extract_windows(
			entry['calcium'], sum(results['input_mask'][0]))
		entry['inputs'] = results['pca'](entry['inputs'])

	pad_left  = int(where(results['output_mask'][1])[0] + .5)
	pad_right = results['output_mask'].shape[1] - pad_left - 1

	for k, entry in enumerate(data):
		if verbosity > 0:
			print 'Predicting cell {0}...'.format(k)

		predictions = []
		for model in results['models']:
			# compute conditional expectation
			predictions.append(model.predict(entry['inputs']).ravel())
		
		entry['predictions'] = hstack([
			zeros(pad_left),
			mean(predictions, 0),
			zeros(pad_right)]).reshape(1, -1)

		# inputs no longer needed
		del entry['inputs']

	return data



def robust_linear_regression(x, y, num_scales=3, max_iter=1000):
	"""
	Performs linear regression with Gaussian scale mixture residuals. 

	$$y = ax + b + \\varepsilon,$$

	where $\\varepsilon$ is assumed to be Gaussian scale mixture distributed.

	@type  x: array_like
	@param x: list of one-dimensional inputs

	@type  y: array_like
	@param y: list of one-dimensional outputs

	@type  num_scales: int
	@param num_scales: number of Gaussian scale mixture components

	@type  max_iter: int
	@param max_iter: number of optimization steps in parameter search

	@rtype: tuple
	@return: slope and y-intercept
	"""

	x = asarray(x).reshape(1, -1)
	y = asarray(y).reshape(1, -1)

	# preprocess inputs
	m = mean(x)
	s = std(x)

	x = (x - m) / s

	# preprocess outputs using simple linear regression
	C = cov(x, y)
	a = C[0, 1] / C[0, 0]
	b = mean(y) - a * mean(x)

	y = y - (a * x + b)

	# robust linear regression
	model = MCGSM(
		dim_in=1,
		dim_out=1,
		num_components=1,
		num_scales=num_scales,
		num_features=0)

	model.initialize(x, y)
	model.train(x, y, parameters={
		'train_means': True,
		'max_iter': max_iter})

	a = (a + float(model.predictors[0])) / s
	b = (b + float(model.means)) - a * m

	return a, b
