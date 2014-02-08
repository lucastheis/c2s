#!/usr/bin/env python

"""
Searches for a set of hyperparameters which minimizes generalization error.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from numpy import min, max, asarray, array, mean, corrcoef, argmax, arange
from pymc import Beta, Exponential, MCMC
from cmt.utils import random_select
from tools import Experiment
from pickle import load
from calcium import train, predict, preprocess

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--dataset',         '-d', type=str,   required=True)
	parser.add_argument('--num_train',       '-t', type=int,   default=10)
	parser.add_argument('--num_components',  '-c', type=int,   nargs='+', default=[1, 2, 3, 4])
	parser.add_argument('--num_features',    '-f', type=int,   nargs='+', default=[0, 1, 2, 3, 4])
	parser.add_argument('--num_valid',       '-v', type=int,   nargs='+', default=[0, 1, 2, 3, 4, 5])
	parser.add_argument('--var_explained',   '-e', type=float, nargs='+', default=arange(85, 100))
	parser.add_argument('--window_length',   '-w', type=float, nargs='+', default=arange(500, 1500, 100))
	parser.add_argument('--num_models',      '-m', type=int,   default=3)
	parser.add_argument('--keep_all',        '-k', type=int,   default=0)
	parser.add_argument('--finetune',        '-n', type=int,   default=1)
	parser.add_argument('--min_repetitions', '-r', type=int,   default=2)
	parser.add_argument('--max_repetitions', '-s', type=int,   default=20)
	parser.add_argument('--max_iter',        '-i', type=int,   default=10)
	parser.add_argument('--alpha',           '-a', type=float, default=.05)
	parser.add_argument('--preprocess',      '-p', type=int,   default=0)

	args = parser.parse_args(argv[1:])

	# load data
	with open(args.dataset) as handle:
		data = load(handle)

	if not 0. < args.alpha < .5:
		raise ValueError('Alpha should be between 0.0 and 0.5.')

	# check validity of hyperparameters
	if args.num_train > len(data) - 1:
		raise ValueError('The number of training cells ({0}) has to be smaller than the size of the dataset ({1}).'.format(
			args.num_train, len(data)))

	if max(args.num_valid) > args.num_train - 1:
		raise ValueError('The number of validation cells ({0}) has to be smaller than the total number of training cells ({1}).'.format(
			max(args.num_valid), args.num_train))

	if args.preprocess:
		data = preprocess(data)

	# initial parameters
	params = {
		'verbosity': 0,
		'num_valid': 0,
		'num_models': args.num_models,
		'keep_all': args.keep_all,
		'finetune': args.finetune,
		'var_explained': 90.,
		'window_length': 800.,
		'model_parameters': {
				'num_components': 1,
				'num_features': 0
			},
		'training_parameters': {
				'verbosity': 0,
				'threshold': 1e-12
			}
		}

	def update_params(key, value):
		if key in ['num_components', 'num_features']:
			params['model_parameters'][key] = value
		else:
			params[key] = value

	for i in range(args.max_iter):
		for key in ['window_length', 'var_explained', 'num_components', 'num_features', 'num_valid']:
			performance = [[] for _ in args.__dict__[key]]

			for _ in range(args.min_repetitions):
				# pick training cells at random
				training_cells = random_select(args.num_train, len(data))

				# split dataset intro training and test set
				data_train = [entry for i, entry in enumerate(data) if i in training_cells]
				data_test = [entry for i, entry in enumerate(data) if i not in training_cells]

				for k, value in enumerate(args.__dict__[key]):
					print 'Testing {0} = {1}...'.format(key, value)

					# set parameter
					update_params(key, value)

					# train on training set and predict spikes on test set
					data_test = predict(data_test, train(data_train, **params), verbosity=0)

					# compute average correlation of predictions with true spikes
					corr = []
					for entry in data_test:
						corr.append(corrcoef(entry['predictions'], entry['spikes'])[0, 1])
					performance[k].append(mean(corr))

				for k, perf in enumerate(performance):
					print '{0} = {1}: '.format(key, args.__dict__[key][k]),
					for val in perf:
						print '{0:.3f} '.format(val),
					print

			# for each parameter, compute probability that expected performance is maximal
			prob, _ = max_corr(performance)

			for _ in range(args.max_repetitions - args.min_repetitions):
				if any(array(prob) > 1. - args.alpha):
					# we have a winner
					break

				# pick training cells at random
				training_cells = random_select(args.num_train, len(data_train))

				# split dataset intro training and test set
				data_train = [entry for i, entry in enumerate(data) if i in training_cells]
				data_test = [entry for i, entry in enumerate(data) if i not in training_cells]

				# confidence not yet high enough, perform another round
				for k, value in enumerate(args.__dict__[key]):
					# set parameter
					update_params(key, value)

					# train on training set and predict spikes on test set
					data_test = predict(data_test, train(data_train, **params), verbosity=0)

					# compute average correlation of predictions with true spikes
					corr = []
					for entry in data_test:
						corr.append(corrcoef(entry['predictions'], entry['spikes'])[0, 1])
					performance[k].append(mean(corr))

				print key
				print args.__dict__[key]
				for perf in performance:
					for val in perf:
						print '{0:.3f} '.format(val),
					print

			# for each parameter, compute probability that expected performance is maximal
			prob, _ = max_corr(performance)

			# pick parameter which most likely yields best expected performance
			update_params(key, args.__dict__[key][argmax(prob)])

			print 'Current parameters:'
			for key in ['window_length', 'var_explained', 'num_components', 'num_features', 'num_valid']:
				print '{0} = {1}'.format(key, params[key])

	return 0



def max_corr(values, num_samples=10000):
	"""
	Estimates the probability that the expected correlation of one sample
	is at least as large as the expected correlations of the other samples.

	This function assumes that all correlations are non-negative.

	Example:

		>>> prob, samples = maxCorr([[0.7, 0.64], [0.6, 0.62, 0.57]])

	@type  values: list
	@param values: list of samples

	@type  num_samples: int
	@param num_samples: number of samples used in approximation of probabilities

	@rtype: tuple
	@return: list of probabilities and list of samples of expected correlation
	"""

	if len(values) < 2:
		return [1.]

	# posterior samples of mean correlations
	samples = []

	for val in values:
		val = asarray(val)
		val[val < 0.] = 1e-16

		# defines prior on mean correlation
		alpha = Exponential('aplha', .1)
		beta = Exponential('beta', .1)

		# observed 
		gamma = Beta('gamma', alpha, beta, observed=True, value=val)

		# sample alpha and beta from posterior
		mcmc = MCMC([alpha, beta, gamma])
		mcmc.sample(iter=num_samples + 1000, burn=1000, progress_bar=False)

		# compute mean correlations
		a = mcmc.trace(alpha)[:]
		b = mcmc.trace(beta)[:]
		samples.append(a / (a + b))

	prob = []

	for i in range(len(values)):
		Ri = samples[i]
		maxRj = max(array([s for j, s in enumerate(samples) if j != i]), 0)

		# estimate probability that correlation is at least as large as
		# all other correlations
		prob.append(mean(Ri >= maxRj))

	return prob, samples



if __name__ == '__main__':
	sys.exit(main(sys.argv))
