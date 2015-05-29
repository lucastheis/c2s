#!/usr/bin/env python

"""
Evaluates firing rate predictions in terms of correlations or Poisson likelihoods.

Examples:

	c2s evaluate data.preprocessed.pck predictions.pck
"""

import os
import sys

from argparse import ArgumentParser
from scipy.io import savemat
from pickle import load
from numpy import mean, min, hstack, asarray
from c2s import evaluate, load_data
from c2s.experiment import Experiment

def print_traces(result, fps):
	for k, r in enumerate(result):
		print '{0:>5} {1:>6.1f} {2:>8.3f}'.format(k, fps[-1][k], r)

	print '-------------------------'
	print '{0:>5} {1:>6.1f} {2:>8.3f}'.format('Avg.', mean(fps[-1]), mean(result))
	print

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('dataset',                 type=str)
	parser.add_argument('predictions',             type=str,   nargs='?')
	parser.add_argument('--downsampling',    '-s', type=int,   default=[1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50], nargs='+')
	parser.add_argument('--optimize',        '-z', type=int,   default=1,
		help='Whether or not to optimize point-wise nonlinearity when evaluating likelihood.')
	parser.add_argument('--regularization',  '-r', type=float, default=5e-8,
		help='Controls smoothness of optimized nonlinearity (default: 5e-8).')
	parser.add_argument('--method',          '-m', type=str,   default='corr', choices=['corr', 'auc', 'info'])
	parser.add_argument('--output',          '-o', type=str,   default='')
	parser.add_argument('--verbosity',       '-v', type=int,   default=1)

	args, _ = parser.parse_known_args(argv[1:])

	experiment = Experiment()

	data = load_data(args.dataset)

	if not args.predictions:
		# use raw calcium signal for prediction
		calcium_min = min(hstack(entry['calcium'] for entry in data))
		for entry in data:
			entry['predictions'] = entry['calcium'] - calcium_min + 1e-5

	else:
		predictions = load_data(args.predictions)

		try:
			if len(predictions) != len(data):
				raise ValueError()

			for entry1, entry2 in zip(data, predictions):
				if entry1['calcium'].size != entry2['predictions'].size:
					raise ValueError()
				entry1['predictions'] = entry2['predictions']

		except ValueError:
			print 'These predictions seem to be for a different dataset.'
			return 1

	fps = []
	loglik = []
	correlations = []
	auc = []
	entropy = []
	functions = []


	for ds in args.downsampling:
		if args.verbosity > 0:
			if args.method.lower().startswith('c'):
				print '{0:>5} {1:>6} {2}'.format('Trace', 'FPS ', 'Correlation')
			elif args.method.lower().startswith('a'):
				print '{0:>5} {1:>6} {2}'.format('Trace', 'FPS ', 'AUC')
			else:
				print '{0:>5} {1:>6} {2}'.format('Trace', 'FPS ', 'Information gain')

		fps.append([])
		for entry in data:
			fps[-1].append(entry['fps'] / ds)

		if args.method.lower().startswith('c'):
			# compute correlations
			R = evaluate(data, method=args.method,
				optimize=args.optimize,
				downsampling=ds,
				verbosity=args.verbosity)

			correlations.append(R)

			if args.verbosity > 0:
				print_traces(R, fps)

		elif args.method.lower().startswith('a'):
			# compute correlations
			A = evaluate(data, method=args.method,
				optimize=args.optimize,
				downsampling=ds,
				verbosity=args.verbosity)

			auc.append(A)

			if args.verbosity > 0:
				print_traces(A, fps)

		else:
			# compute log-likelihoods
			L, H, f = evaluate(data, method='loglik',
				optimize=args.optimize,
				downsampling=ds,
				verbosity=args.verbosity,
				return_all=True,
				regularize=args.regularization)

			loglik.append(L)
			entropy.append(H)
			functions.append((f.x, f.y))

			if args.verbosity > 0:
				print_traces(H + L, fps)

	if args.output.lower().endswith('.mat'):

		if args.method.lower().startswith('c'):
			savemat(args.output, {'fps': asarray(fps), 'correlations': asarray(correlations)})
		elif args.method.lower().startswith('a'):
			savemat(args.output, {'fps': asarray(fps), 'auc': asarray(auc)})
		else:
			savemat(args.output, {
				'fps': asarray(fps),
				'loglik': asarray(loglik),
				'entropy': asarray(entropy),
				'info': asarray(loglik) + asarray(entropy)})

	elif args.output:
		if os.path.isdir(args.output):
			filepath = os.path.join(args.output, args.method + '.{0}.{1}.xpck')
		else:
			filepath = args.output

		experiment['args'] = args
		experiment['fps'] = asarray(fps)

		if args.method.lower().startswith('c'):
			experiment['correlations'] = asarray(correlations)
		elif args.method.lower().startswith('a'):
			experiment['auc'] = asarray(auc)
		else:
			experiment['loglik'] = asarray(loglik)
			experiment['entropy'] = asarray(entropy)
			experiment['info'] = asarray(loglik) + asarray(entropy)
			experiment['f'] = functions

		experiment.save(filepath, overwrite=True)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
