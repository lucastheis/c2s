"""
Tools for the prediction of spike trains from calcium traces.
"""

from copy import copy, deepcopy
from numpy import percentile, asarray, arange, zeros, where, repeat, sort, cov, mean, std, ceil
from numpy import vstack, hstack, argmin, ones, convolve, log, linspace, min, max, square, sum, diff
from numpy import corrcoef, array, eye, dot, empty_like
from numpy.random import rand
from scipy.signal import resample
from scipy.stats import poisson
from scipy.stats.mstats import gmean
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from cmt.models import MCGSM, STM, Poisson
from cmt.nonlinear import ExponentialFunction, BlobNonlinearity
from cmt.tools import generate_data_from_image, extract_windows
from cmt.transforms import PCATransform
from cmt.utils import random_select

try:
	from roc import roc
except:
	pass

def preprocess(data, fps=100., filter=None, verbosity=0):
	"""
	Normalize calcium traces and spike trains.

	This function does three things:
		1. Remove any linear trends using robust linear regression.
		2. Normalize the range of the calcium trace by the 5th and 80th percentile.
		3. Change the sampling rate of the calcium trace and spike train.

	If `filter` is set, the first step is replaced by estimating and removing a baseline using
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

	@type  regularize: float
	@param regularize: strength with which model filters are regularized

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



def predict(data, results, max_spikes_per_sec=1000., verbosity=1):
	"""
	Predicts firing rates from calcium traces using spiking neuron models.

	@type  data: list
	@param data: list of dictionaries containing calcium/fluorescence traces

	@type  results: dict
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
	@param data: list of dictionaries as produced by L{predict}

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

			# compute area under curve
			auc.append(roc(pos, neg)[0])

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
