"""
Tools for the prediction of spike trains from calcium traces.

This module contains functions for predicting spikes from fluorescence traces obtained
from two-photon calcium images. Data should be stored as a list of dictionaries, where
each dictionary corresponds to a cell or recording. Each dictionary has to contain at least
the entries C{calcium} and C{fps}, which correspond to the recorded fluorescence trace and
its sampling rate in frames per second.

	>>> data = [
	>>>	{'calcium': [[0., 0., 0., 0.]],     'fps': 10.4},
	>>>	{'calcium': [[0., 0., 0., 0., 0.]], 'fps': 12.1}]

The data here is only used to illustrate the format. Each calcium trace is expected to
be given as a 1xT array, where T is the number of recorded frames. After importing the
module,

	>>> import c2s

we can use L{preprocess<c2s.preprocess>} to normalize the calcium traces and
C{predict<c2s.predict>} to predict firing rates:

	>>> data = c2s.preprocess(data)
	>>> data = c2s.predict(data)

The predictions for the i-th cell can be accessed via:

	>>> data[i]['predictions']

Simultaneously recorded spikes can be stored either as binned traces

	>>> data = [
	>>>	{'calcium': [[0., 0., 0., 0.]],     'spikes': [[0, 1, 0, 2]],    'fps': 10.4},
	>>>	{'calcium': [[0., 0., 0., 0., 0.]], 'spikes': [[0, 0, 3, 1, 0]], 'fps': 12.1}]

or, preferably, as spike times in milliseconds:

	>>> data = [
	>>>	{'calcium': [[0., 0., 0., 0.]],     'spike_times': [[15.1, 35.2, 38.1]],      'fps': 10.4},
	>>>	{'calcium': [[0., 0., 0., 0., 0.]], 'spike_times': [[24.2, 28.4 32.7, 40.2]], 'fps': 12.1}]

The preprocessing function will automatically compute the other format of the spike trains if one
of them is given. Using the method L{train<c2s.train>}, we can train a model to predict spikes from
fluorescence traces

	>>> data = c2s.preprocess(data)
	>>> results = c2s.train(data)

and then use it to make predictions:

	>>> data = c2s.predict(data, results)

It is important that the data used for training undergoes the same preprocessing as the data
used when making predictions.

@undocumented: optimize_predictions
@undocumented: robust_linear_regression
@undocumented: percentile_filter
@undocumented: downsample
@undocumented: responses
@undocumented: generate_inputs_and_outputs
@undocumented: DEFAULT_MODEL
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'
__version__ = '0.1.0dev'

import sys
from copy import copy, deepcopy
from base64 import b64decode
from pickle import load, loads
from numpy import percentile, asarray, arange, zeros, where, repeat, sort, cov, mean, std, ceil
from numpy import vstack, hstack, argmin, ones, convolve, log, linspace, min, max, square, sum, diff
from numpy import corrcoef, array, eye, dot, empty_like
from numpy.random import rand
from scipy.signal import resample
from scipy.stats import poisson
from scipy.stats.mstats import gmean
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.io import loadmat
from cmt.models import MCGSM, STM, Poisson
from cmt.nonlinear import ExponentialFunction, BlobNonlinearity
from cmt.tools import generate_data_from_image, extract_windows
from cmt.transforms import PCATransform
from cmt.utils import random_select
from experiment import Experiment

try:
	from roc import roc
except:
	pass

def load_data(filepath):
	"""
	Loads data in either pickle or MATLAB format.

	@type  filepath: string
	@param filepath: path to dataset

	@rtype: list
	@return: list of dictionaries containing the data
	"""

	if filepath.lower().endswith('.mat'):
		data = []
		data_mat = loadmat(filepath)
		data_mat = data_mat['data'].ravel()

		for entry_mat in data_mat:
			entry = {}

			for key in entry_mat.dtype.names:
				entry[key] = entry_mat[key][0, 0]

			for key in ['calcium', 'spikes', 'spike_times']:
				if key in entry:
					entry[key] = entry[key].reshape(1, entry[key].size)
			if 'fps' in entry:
				entry['fps'] = float(entry['fps'])
			if 'cell_num' in entry:
				entry['cell_num'] = int(entry['cell_num'])

			data.append(entry)

		return data

	if filepath.lower().endswith('.xpck'):
		return Experiment(filepath)['data']

	with open(filepath) as handle:
		return load(handle)



def preprocess(data, fps=100., filter=None, verbosity=0):
	"""
	Normalize calcium traces and spike trains.

	This function does three things:
		1. Remove any linear trends using robust linear regression.
		2. Normalize the range of the calcium trace by the 5th and 80th percentile.
		3. Change the sampling rate of the calcium trace and spike train.

	If C{filter} is set, the first step is replaced by estimating and removing a baseline using
	a percentile filter (40 seconds seems like a good value for the percentile filter).

	@type  data: list
	@param data: list of dictionaries containing calcium/fluorescence traces

	@type  fps: float
	@param fps: desired sampling rate of signals

	@type  filter: float/None
	@param filter: number of seconds used in percentile filter

	@type  verbosity: int
	@param verbosity: if positive, print messages indicating progress

	@rtype: list
	@return: list of preprocessed recordings
	"""

	data = deepcopy(data)

	for k in range(len(data)):
		if verbosity > 0:
			print 'Preprocessing calcium trace {0}...'.format(k)

		data[k]['fps'] = float(data[k]['fps'])

		if filter is None:
			# remove any linear trends
			x = arange(data[k]['calcium'].size)
			a, b = robust_linear_regression(x, data[k]['calcium'])

			data[k]['calcium'] = data[k]['calcium'] - (a * x + b)
		else:
			data[k]['calcium'] = data[k]['calcium'] - \
				percentile_filter(data[k]['calcium'], window_length=int(data[k]['fps'] * filter), perc=5)

		# normalize dispersion
		calcium05 = percentile(data[k]['calcium'],  5)
		calcium80 = percentile(data[k]['calcium'], 80)

		if calcium80 - calcium05 > 0.:
			data[k]['calcium'] = (data[k]['calcium'] - calcium05) / float(calcium80 - calcium05)

		# compute spike times if binned spikes are given
		if 'spikes' in data[k] and 'spike_times' not in data[k]:
			spikes = asarray(data[k]['spikes'].ravel(), dtype='uint16')

			# compute spike times in milliseconds
			spike_times = where(spikes > 0)[0]
			spike_times = repeat(spike_times, spikes[spike_times])
			spike_times = (spike_times + rand(*spike_times.shape)) * (1000. / data[k]['fps'])

			data[k]['spike_times'] = sort(spike_times).reshape(1, -1)

		if fps is not None and fps > 0.:
			# number of samples after update of sampling rate
			num_samples = int(float(data[k]['calcium'].size) * fps / data[k]['fps'] + .5)

			if num_samples != data[k]['calcium'].size:
				# factor by which number of samples will actually be changed
				factor = num_samples / float(data[k]['calcium'].size)

				# resample calcium signal
				data[k]['calcium'] = resample(data[k]['calcium'].ravel(), num_samples).reshape(1, -1)
				data[k]['fps'] = data[k]['fps'] * factor
		else:
			# don't change sampling rate
			num_samples = data[k]['calcium'].size

		# compute binned spike trains if missing
		if 'spike_times' in data[k] and ('spikes' not in data[k] or num_samples != data[k]['spikes'].size):
			# spike times in bins
			spike_times = asarray(data[k]['spike_times'] * (data[k]['fps'] / 1000.), dtype=int).ravel()
			spike_times = spike_times[spike_times < num_samples]
			spike_times = spike_times[spike_times >= 0]

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
		training_parameters={},
		regularize=0.):
	"""
	Trains models on the task of predicting spike trains from calcium traces.

	This function takes a dataset and trains one or several models (STMs) to predict spikes
	from calcium signals. By default, the method trains a single model on 1000ms windows
	extracted from the calcium (fluorescence) traces.

		>>> results = train(data)

	See above for an explanation of the expected data format. A more detailed example:

		>>> results = train(data,
		>>>	num_models=4,
		>>>	var_explained=98.,
		>>>	window_length=800.,
		>>>	model_parameters={
		>>>		'num_components': 3,
		>>>		'num_features': 2},
		>>>	training_parameters={
		>>>		'max_iter': 3000,
		>>>		'threshold': 1e-9})

	For an explanation on the model and training parameters, please see the
	U{CMT documentation<http://lucastheis.github.io/cmt/>}. The training procedure
	returns a dictionary containing the trained models, and things needed for handling
	and preprocessing calcium traces.

		>>> results['models']
		>>> results['pca']
		>>> results['input_mask']
		>>> results['output_mask']

	@see: L{predict<c2s.predict>}

	@type  data: list
	@param data: list of dictionaries containig calcium/fluorescence traces

	@type  num_valid: int
	@param num_valid: number of cells used for early stopping based on a validation set
	
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

	@type  regularize: float
	@param regularize: strength with which model filters are regularized for smoothness

	@rtype: dict
	@return: dictionary containing trained models and things needed for preprocessing
	"""

	model_parameters.setdefault('num_components', 3)
	model_parameters.setdefault('num_features', 2)
	model_parameters.setdefault('nonlinearity', ExponentialFunction)
	model_parameters.setdefault('distribution', Poisson)

	training_parameters.setdefault('max_iter', 3000)
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
			dim_in_nonlinear=pca.dim_in_pre,
			dim_in_linear=0,
			**model_parameters)

		if num_valid > 0:
			idx = random_select(num_valid, len(data))

			inputs_train  = hstack(entry['inputs']  for k, entry in enumerate(data) if k in idx)
			inputs_valid  = hstack(entry['inputs']  for k, entry in enumerate(data) if k not in idx)
			outputs_train = hstack(entry['outputs'] for k, entry in enumerate(data) if k in idx)
			outputs_valid = hstack(entry['outputs'] for k, entry in enumerate(data) if k not in idx)

			inputs_outputs = (inputs_train, outputs_train, inputs_valid, outputs_valid)
		else:
			inputs  = hstack(entry['inputs'] for entry in data)
			outputs = hstack(entry['outputs'] for entry in data)

			inputs_outputs = (inputs, outputs)

		if regularize > 0.:
			transform = eye(pca.dim_in) \
				- eye(pca.dim_in, pca.dim_in,  1) / 2. \
				- eye(pca.dim_in, pca.dim_in, -1) / 2.
			transform = dot(transform, pca.pre_in.T)

			training_parameters['regularize_predictors'] = {
				'strength': regularize,
				'transform': transform,
				'norm': 'L1'}
			training_parameters['regularize_features'] = {
				'strength': regularize / 10.,
				'transform': transform,
				'norm': 'L1'}
			
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



def predict(data, results=None, max_spikes_per_sec=1000., verbosity=1):
	"""
	Predicts firing rates from calcium traces using spiking neuron models.

	If no model is specified via C{results}, a default model is used which was trained
	on two datasets of V1 recordings of mice (dataset 1 and 2 of Theis et al., 2014).

	@type  data: list
	@param data: list of dictionaries containing calcium/fluorescence traces

	@type  results: dict/None
	@param results: dictionary containing results of training procedure

	@type  max_spikes_per_sec: float
	@param max_spikes_per_sec: prevents unreasonable spike count predictions

	@type  verbosity: int
	@param verbosity: if positive, print messages indicating progress

	@rtype: list
	@return: returns a list of dictionaries like C{data} but with added predictions
	"""

	if type(data) is dict:
		data = [data]
	if type(data) is not list or (len(data) > 0 and type(data[0]) is not dict):
		data = [{'calcium': data}]
	if results is None:
		results = loads(b64decode(DEFAULT_MODEL))

	# create copies of dictionaries (doesn't create copies of actual data arrays)
	data = [copy(entry) for entry in data]

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

		max_spikes = max_spikes_per_sec / float(entry['fps'])

		predictions = []
		for model in results['models']:
			# compute conditional expectation
			pred = model.predict(entry['inputs']).ravel()
			pred[pred > max_spikes] = max_spikes
			predictions.append(pred)

		# average predicted firing rate
		avg = mean(asarray(gmean(predictions, 0)))

		entry['predictions'] = hstack([
			zeros(pad_left) + avg,
			asarray(gmean(predictions, 0)),
			zeros(pad_right) + avg]).reshape(1, -1)

		# inputs no longer needed
		del entry['inputs']

	return data



def evaluate(data, method='corr', **kwargs):
	"""
	Evaluates predictions using either Pearson's correlation, log-likelihood, information rates,
	or area under the ROC curve.

	@type  data: list
	@param data: list of dictionaries as produced by L{predict<c2s.predict>}

	@type  method: string
	@param method: either 'loglik', 'info', 'corr', or 'auc' (default: 'corr')

	@type  downsampling: int
	@param downsampling: downsample spike trains and predictions by this factor before evaluation (default: 1)

	@type  optimize: bool
	@param optimize: find optimal point-wise monotonic nonlinearity before evaluation of log-likelihood (default: True)

	@type  regularize: int
	@param regularize: regularize point-wise monotonic nonlinearity to be smooth (default: 5e-3)

	@type  verbosity: int
	@param verbosity: controls output during optimization of nonlinearity (default: 2)

	@type  return_all: bool
	@param return_all: if true, return additional information and not just performance (default: False)

	@rtype: ndarray
	@return: a value for each cell
	"""

	kwargs.setdefault('downsampling', 1)
	kwargs.setdefault('num_support', 10)
	kwargs.setdefault('regularize', 5e-8)
	kwargs.setdefault('optimize', True)
	kwargs.setdefault('return_all', False)
	kwargs.setdefault('verbosity', 2)

	if 'downsample' in kwargs:
		print 'Did you mean `downsampling`?'
		return

	if 'regularization' in kwargs:
		print 'Did you mean `regularize`?'
		return

	if method.lower().startswith('c'):
		corr = []

		# compute correlations
		for entry in data:
			corr.append(
				corrcoef(
					downsample(entry['predictions'], kwargs['downsampling']),
					downsample(entry['spikes'], kwargs['downsampling']))[0, 1])

		return array(corr)

	elif method.lower().startswith('a'):
		auc = []

		# compute area under ROC curve
		for entry in data:
			# downsample firing rates
			predictions = downsample(entry['predictions'], kwargs['downsampling']).ravel()
			spikes = array(downsample(entry['spikes'], kwargs['downsampling']).ravel() + .5, dtype=int)

			# marks bins containing spikes
			mask = spikes > .5

			# collect positive and negative examples
			neg = predictions[-mask]
			pos = []

			# this loop is necessary because any bin can contain more than one spike
			while any(mask):
				pos.append(predictions[mask])
				spikes -= 1
				mask = spikes > .5
			pos = hstack(pos)

			try:
				# compute area under curve
				auc.append(roc(pos, neg)[0])
			except NameError:
				print 'You need to compile `roc.pyx` with cython first.'
				sys.exit(1)

		return array(auc)

	else:
		# downsample predictions and spike trains
		spikes = [downsample(entry['spikes'], kwargs['downsampling']) for entry in data]
		predictions = [downsample(entry['predictions'], kwargs['downsampling']) for entry in data]

		if kwargs['optimize']:
			# find optimal point-wise monotonic function
			f = optimize_predictions(
				hstack(predictions),
				hstack(spikes),
				kwargs['num_support'],
				kwargs['regularize'],
				kwargs['verbosity'])
		else:
			f = lambda x: x
			f.x = [min(hstack(predictions)), max(hstack(predictions))]
			f.y = f.x

		# for conversion into bit/s
		factor = 1. / kwargs['downsampling'] / log(2.)

		# average firing rate (Hz) over all cells
		firing_rate = mean(hstack([s * data[k]['fps'] for k, s in enumerate(spikes)]))

		# estimate log-likelihood and marginal entropies
		loglik, entropy = [], []
		for k in range(len(data)):
			loglik.append(mean(poisson.logpmf(spikes[k], f(predictions[k]))) * data[k]['fps'] * factor)
			entropy.append(-mean(poisson.logpmf(spikes[k], firing_rate / data[k]['fps'])) * data[k]['fps'] * factor)

		if method.lower().startswith('l'):
			if kwargs['return_all']:
				return array(loglik), array(entropy), f
			else:
				# return log-likelihood
				return array(loglik)
		else:
			# return information rates
			return array(loglik) + array(entropy)



def optimize_predictions(predictions, spikes, num_support=10, regularize=5e-8, verbosity=1):
	"""
	Fits a monotonic piecewise linear function to maximize the Poisson likelihood of
	firing rate predictions interpreted as Poisson rate parameter.

	@type  predictions: array_like
	@param predictions: predicted firing rates

	@type  spikes: array_like
	@param spikes: true spike counts

	@type  num_support: int
	@param num_support: number of support points of the piecewise linear function

	@type  regularize: float
	@param regularize: strength of regularization for smoothness

	@rtype: interp1d
	@return: a piecewise monotonic function
	"""

	if num_support < 2:
		raise ValueError('`num_support` should be at least 2.')

	# support points of piece-wise linear function
	if num_support > 2:
		F = predictions
		F = F[F > (max(F) - min(F)) / 100.]
		x = list(percentile(F, range(0, 101, num_support)[1:-1]))
		x = asarray([0] + x + [max(F)])
	else:
		x = asarray([min(predictions), max(predictions)])

	def objf(y):
		# construct piece-wise linear function
		f = interp1d(x, y)

		# compute predicted firing rates
		l = f(predictions) + 1e-16

		# compute negative log-likelihood (ignoring constants)
		K = mean(l - spikes * log(l))
		
		# regularize curvature
		z = (x[2:] - x[:-2]) / 2.
		K = K + regularize * sum(square(diff(diff(y) / diff(x)) / z))

		return K

	class MonotonicityConstraint:
		def __init__(self, i):
			self.i = i

		def __call__(self, y):
			return y[self.i] - y[self.i - 1]

	# monotonicity and non-negativity constraint
	constraints = [{'type': 'ineq', 'fun': MonotonicityConstraint(i)} for i in range(1, x.size)]
	constraints.extend([{'type': 'ineq', 'fun': lambda y: y[0]}])

	# fit monotonic function
	res = minimize(
		fun=objf,
		x0=x + 1e-6,
		method='SLSQP',
		tol=1e-9,
		constraints=constraints,
		options={'disp': 1, 'iprint': verbosity})

	# construct monotonic piecewise linear function
	return interp1d(x, res.x, bounds_error=False, fill_value=res.x[-1])



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



def percentile_filter(x, window_length, perc=5):
	"""
	For each point in a signal, computes a percentile from a window surrounding it.
	"""

	y = empty_like(x)
	d = window_length // 2 + 1

	for t in range(len(x)):
		fr = max([t - d + 1, 0])
		to = t + d
		y[t] = percentile(x[fr:to], perc)

	return y



def downsample(signal, factor):
	"""
	Downsample signal by averaging neighboring values.

	@type  signal: array_like
	@param signal: one-dimensional signal to be downsampled

	@type  factor: int
	@param factor: this many neighboring values are averaged

	@rtype: ndarray
	@return: downsampled signal
	"""

	if factor < 2:
		return asarray(signal).ravel()
	return convolve(asarray(signal).ravel(), ones(factor), 'valid')[::factor]



def responses(data, results, verbosity=0):
	"""
	Compute nonlinear component responses of STM to calcium.

	@type  data: list
	@param data: list of dictionaries containing calcium/fluorescence traces

	@type  results: dict
	@param results: dictionary containing results of training procedure

	@type  verbosity: int
	@param verbosity: if positive, print messages indicating progress

	@rtype: list
	@return: list of dictionaries as input, but with added responses
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
			print 'Computing responses for cell {0}...'.format(k)

		# pick first model
		if type(results) is STM:
			model = results
		else:
			model = results['models'][0]

		# compute nonlinear component responses
		responses = model.nonlinear_responses(entry['inputs'])

		entry['responses'] = hstack([
			zeros([responses.shape[0], pad_left]),
			responses,
			zeros([responses.shape[0], pad_right])])

		# inputs no longer needed
		del entry['inputs']

	return data



def generate_inputs_and_outputs(data, var_explained=95., window_length=1000., pca=None, verbosity=1):
	"""
	Extracts input and output windows from calcium and spike traces.

	@type  data: list
	@param data: list of dictionaries containig calcium/fluorescence traces
	
	@type  var_explained: float
	@param var_explained: controls the number of principal components used to represent calcium window

	@type  window_length: int
	@param window_length: size of calcium window used as input to STM (in milliseconds)

	@type  pca: PCATransform
	@param pca: if given, use results of previous PCA

	@rtype: tuple
	@return: inputs, outputs and a dictionary containing window masks and PCA results
	"""

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

	if pca is None:
		inputs = hstack(entry['inputs'] for entry in data)

		if verbosity > 0:
			print 'Performing PCA...'

		pca = PCATransform(inputs, var_explained=var_explained)

	if verbosity > 0:
		print 'Reducing dimensionality of data...'

	for entry in data:
		entry['inputs'] = pca(entry['inputs'])

	inputs = hstack(entry['inputs'] for entry in data)
	outputs = hstack(entry['outputs'] for entry in data)

	return inputs, outputs, {
		'input_mask': input_mask,
		'output_mask': output_mask,
		'pca': pca}



# base64 encoded pickled default model
DEFAULT_MODEL = "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"
