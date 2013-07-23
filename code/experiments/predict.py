#!/usr/bin/env python

"""
Train a spike-triggered mixture model (STM) to predict spikes from calcium traces.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from itertools import chain
from numpy import log, int, double, hstack, vstack, zeros, sum, mean, corrcoef
from numpy import asarray, where
from numpy.linalg import norm
from numpy.random import permutation
from scipy.io import loadmat, savemat
from cmt.models import STM, GLM, Poisson
from cmt.nonlinear import ExponentialFunction
from cmt.tools import generate_data_from_image
from cmt.transforms import WhiteningTransform
from cmt.utils import random_select
from tools import Experiment

input_mask = asarray([
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype='bool')
output_mask = asarray([
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype='bool')

# cells used for training
cells = [5, 7, 48, 19, 50, 12, 17, 18, 28]

def main(argv):
	experiment = Experiment()

	# parse input arguments
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('-n', '--num_components', type=int, default=2)
	parser.add_argument('-f', '--num_features', type=int, default=5)
	parser.add_argument('-v', '--num_valid', type=int, default=3)
	parser.add_argument('-s', '--num_samples', type=int, default=100)

	args = parser.parse_args(argv[1:])



	### DATA HANDLING

	print 'Preprocessing...'

	# load data
	data = loadmat('data/data.mat')['Data'].ravel()

	# generate inputs and outputs for training and testing
	inputs_train  = []
	inputs_valid  = []
	inputs_test   = []
	outputs_train = []
	outputs_valid = []
	outputs_test  = []

	# split cells into training and validation set
	idx = random_select(args.num_valid, len(cells))
	validation_cells = [c for i, c in enumerate(cells) if i in idx]
	training_cells = [c for i, c in enumerate(cells) if i not in idx]

	for i in range(data.shape[0]):
		galvo_traces = data[i][0].reshape(1, -1)
		spike_traces = data[i][1].reshape(1, -1)

		# extract windows
		inputs, outputs = generate_data_from_image(
			vstack([galvo_traces, spike_traces]), input_mask, output_mask)

		if i + 1 in training_cells:
			inputs_train.append(inputs)
			outputs_train.append(outputs)
		elif i + 1 in validation_cells:
			inputs_valid.append(inputs)
			outputs_valid.append(outputs)
		else:
			inputs_test.append(inputs)
			outputs_test.append(outputs)

	data_train = hstack(inputs_train), hstack(outputs_train)
	data_valid = hstack(inputs_valid), hstack(outputs_valid)
	data_test  = hstack(inputs_test),  hstack(outputs_test)


	# preprocessing of input data
	pre = WhiteningTransform(*data_train)



	### MODEL FITTING

	print 'Training...'

	# choose and fit neuron model
	if args.num_components > 1 or args.num_features > 0:
		model = STM(
			dim_in_nonlinear=sum(input_mask[0]),
			dim_in_linear=sum(input_mask[1]),
			num_components=args.num_components,
			num_features=args.num_features,
			nonlinearity=ExponentialFunction,
			distribution=Poisson)
	else:
		model = GLM(sum(input_mask), ExponentialFunction, Poisson)

	model.train(*chain(pre(*data_train), pre(*data_valid)), parameters={
		'verbosity': 1,
		'max_iter': 1000,
		'val_iter': 1,
		'val_look_ahead': 100,
		'threshold': 1e-9})



	### PREDICTION

	print 'Predicting...'

	# generate sample spike trains
	predictions       = []
	predictions_train = []
	predictions_valid = []
	predictions_test  = []

	outputs       = []
	outputs_train = []
	outputs_valid = []
	outputs_test  = []

	pad_left  = int(where(output_mask[1])[0] + .5)
	pad_right = output_mask.shape[1] - pad_left - 1

	for i in range(data.shape[0]):
		# pick preprocessed inputs
		if i + 1 in training_cells:
			inputs = inputs_train[0]
			inputs_train = inputs_train[1:]
		elif i + 1 in validation_cells:
			inputs = inputs_valid[0]
			inputs_valid = inputs_valid[1:]
		else:
			inputs = inputs_test[0]
			inputs_test = inputs_test[1:]

		# sample responses
		predictions_ = []
		for j in range(args.num_samples):
			predictions_.append(model.sample(pre(inputs)))
		predictions_ = vstack(predictions_)

		# average responses
		predictions_ = hstack([
			zeros(pad_left),
			mean(predictions_, 0),
			zeros(pad_right)]).reshape(1, -1)

		# store predictions
		predictions.append(predictions_)
		outputs.append(data[i][1].reshape(1, -1))
		if i + 1 in training_cells:
			predictions_train.append(predictions[-1])
			outputs_train.append(outputs[-1])
		elif i + 1 in validation_cells:
			predictions_valid.append(predictions[-1])
			outputs_valid.append(outputs[-1])
		else:
			predictions_test.append(predictions[-1])
			outputs_test.append(outputs[-1])



	### EVALUATION

	print 'Evaluating...'

	# average log-likelihood in bits per bin
	loglik_train = mean(model.loglikelihood(*pre(*data_train)))
	loglik_valid = mean(model.loglikelihood(*pre(*data_valid)))
	loglik_test  = mean(model.loglikelihood(*pre(*data_test)))

	# average sampling rate
	sampling_rate = mean([data[i][2] for i in range(data.shape[0])])

	# compute correlation
	corr       = corrcoef(hstack(outputs),       hstack(predictions))[0, 1]
	corr_train = corrcoef(hstack(outputs_train), hstack(predictions_train))[0, 1]
	corr_valid = corrcoef(hstack(outputs_valid), hstack(predictions_valid))[0, 1]
	corr_test  = corrcoef(hstack(outputs_test),  hstack(predictions_test))[0, 1]

	# print results
	print
	print 'Number of spikes:'
	print '\t{0} (training)'.format(sum(data_train[1]))
	print '\t{0} (validation)'.format(sum(data_valid[1]))
	print '\t{0} (test)'.format(sum(data_test[1]))
	print
	print 'Log-likelihood:'
	print '\t{0:.2f} [bit/s] (training)'.format(loglik_train / log(2.) * sampling_rate)
	print '\t{0:.2f} [bit/s] (validation)'.format(loglik_valid / log(2.) * sampling_rate)
	print '\t{0:.2f} [bit/s] (test)'.format(loglik_test / log(2.) * sampling_rate)
	print
	print 'Correlation:'
	print '\t{0:.5f} (training)'.format(corr_train)
	print '\t{0:.5f} (validation)'.format(corr_valid)
	print '\t{0:.5f} (test)'.format(corr_test)
	print '\t{0:.5f} (total)'.format(corr)

	# save results
	experiment['training_cells']   = training_cells
	experiment['validation_cells'] = validation_cells

	experiment['model'] = model

	experiment['corr']       = corr
	experiment['corr_train'] = corr_train
	experiment['corr_valid'] = corr_valid
	experiment['corr_test']  = corr_test

	experiment['loglik_train'] = loglik_train
	experiment['loglik_valid'] = loglik_valid
	experiment['loglik_test']  = loglik_test

	experiment['predictions']       = predictions
	experiment['predictions_train'] = predictions_train
	experiment['predictions_valid'] = predictions_valid
	experiment['predictions_test']  = predictions_test

	experiment.save('results/predictions.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
