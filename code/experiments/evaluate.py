#!/usr/bin/env python

"""
Evaluates firing rate predictions in terms of correlations or Poisson likelihoods.
"""

import os
import sys

sys.path.append('./code')

from argparse import ArgumentParser
from scipy.io import loadmat
from pickle import load
from numpy import mean, min, hstack
from tools import Experiment
from calcium import evaluate

def main(argv):
	experiment = Experiment()

	parser = ArgumentParser(argv[0], description=__doc__)

	parser.add_argument('--results',      '-r', type=str, default='')
	parser.add_argument('--dataset',      '-d', type=str, default='')
	parser.add_argument('--downsampling', '-s', type=int, default=[1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50], nargs='+')
	parser.add_argument('--optimize',     '-z', type=int, default=1)
	parser.add_argument('--verbosity',    '-v', type=int, default=2)
	parser.add_argument('--method',       '-m', type=str, default='loglik')
	parser.add_argument('--output',       '-o', type=str, default='results/')

	args, _ = parser.parse_known_args(argv[1:])

	if args.results == '':
		# use raw calcium signal for prediction
		with open(args.dataset) as handle:
			data = load(handle)

		calcium_min = min(hstack(entry['calcium'] for entry in data))

		for entry in data:
			entry['predictions'] = entry['calcium'] - calcium_min + 1e-5

	else:
		if args.results.endswith('.mat'):
			# results are stored in MATLAB file
			results = {'predictions': loadmat(args.results)['predictions'].ravel()}

			if args.dataset:
				try:
					with open(args.dataset) as handle:
						data = load(handle)
				except IOError:
					print 'Could not open dataset.'
			else:
				print 'Please specify which dataset corresponds to these predictions.'
		else:
			# results are stored in pickled experiment
			results = Experiment(args.results)

			try:
				if args.dataset:
					with open(args.dataset) as handle:
						data = load(handle)
				else:
					with open(results['args'].dataset) as handle:
						data = load(handle)
			except IOError:
				print 'Could not open dataset.'

		for k, entry in enumerate(data):
			entry['predictions'] = results['predictions'][k]

	fps = []
	loglik = []
	correlations = []
	auc = []
	entropy = []
	functions = []

	# compute likelihood
	for ds in args.downsampling:
		fps.append([])
		for entry in data:
			fps[-1].append(entry['fps'] / ds)

		if args.method.startswith('c'):
			# compute correlations
			R = evaluate(data, method=args.method,
				optimize=args.optimize,
				downsampling=ds,
				verbosity=args.verbosity)

			correlations.append(R)

			for r in R:
				print '{0:.3f}'.format(r)

			print '------'
			print '{0:.3f}'.format(mean(R))
			print

		elif args.method.startswith('a'):
			# compute correlations
			A = evaluate(data, method=args.method,
				optimize=args.optimize,
				downsampling=ds,
				verbosity=args.verbosity)

			auc.append(A)

			for a in A:
				print '{0:.3f}'.format(a)

			print '------'
			print '{0:.3f}'.format(mean(A))
			print

		else:
			# compute log-likelihoods
			L, H, f = evaluate(data, method=args.method,
				optimize=args.optimize,
				downsampling=ds,
				verbosity=args.verbosity,
				return_all=True,
				regularize=5e-8)

			loglik.append(L)
			entropy.append(H)
			functions.append((f.x, f.y))

			print
			for I in (H + L):
				print '{0:.3f} [bit/s]'.format(I)
			print '------'
			print '{0:.3f} [bit/s]'.format(mean(H + L))
			print

	if os.path.isdir(args.output):
		if args.results.endswith('.xpck'):
			filepath = args.results[:-4] + args.method + '.xpck'
		else:
			filepath = os.path.join(args.output, args.method + '.{0}.{1}.xpck')
	else:
		filepath = args.output

	experiment['args'] = args
	experiment['fps'] = fps

	if args.method.startswith('c'):
		experiment['correlations'] = correlations
	elif args.method.startswith('a'):
		experiment['auc'] = auc
	else:
		experiment['loglik'] = loglik
		experiment['entropy'] = entropy
		experiment['f'] = functions

	experiment.save(filepath, overwrite=True)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))

