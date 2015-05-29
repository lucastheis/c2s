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
from numpy import mean, min, hstack, asarray, average, unique
from c2s import evaluate, load_data
from c2s.experiment import Experiment

def print_traces(result, fps):
	"""
	Prints the result for each trace and averages over traces.
	"""
	for k, r in enumerate(result):
		print '{0:>5} {1:>6.1f} {2:>8.3f}'.format(k, fps[-1][k], r)

	print '-------------------------'
	print '{0:>5} {1:>6.1f} {2:>8.3f}'.format('Avg.', mean(fps[-1]), mean(result))
	print

def print_weighted_average(result, data, downsampling):
	"""
	Prints the result for each cell by calculating a weighted average of all traces of a cell.
	The overall average of cells is also weighted by the recording time of the cell.
	"""
	
	if not 'cell_num' in data[0]:
		cell_results = result
		cell_nums = range(len(cell_results))
		cell_fps = asarray([entry['fps'] / downsampling for entry in data])
		cell_weights = [len(entry['calcium']) / entry['fps'] for entry in data]
		weighted_average = average(cell_results, weights=cell_weights)
	else:
		# the following code can be written more efficiently,
		# but it's not necessary given the small number of traces and cells
		
		cell_nums = unique([entry['cell_num'] for entry in data])
		cell_results = []
		cell_fps = []
		cell_weights = []
		for i in cell_nums:
			traces_results = []
			traces_fps = []
			traces_weights = []
			# find the results and weights for all traces belonging to cell i
			for k, entry in enumerate(data):
				if entry['cell_num'] == i:
					traces_results.append(results[i])
					traces_fps.append(entry['fps'] / downsampling)
					traces_weights.append(len(entry['calcium']) / entry['fps'])
			cell_results.append(average(traces_results, weights=traces_weights))
			cell_fps.append(average(traces_fps, weights=traces_weights))
			cell_weights.append(sum(traces_weights))
		
		cell_results = asarray(cell_results)
		cell_fps = asarray(cell_fps)
		weighted_average = average(cell_results, weights=cell_weights)
	
	for k, r in enumerate(cell_results):
		print '{0:>5} {1:>6.1f} {2:>8.3f}'.format(cell_nums[k], cell_fps[k], r)

	print '-------------------------'
	print '{0:>5} {1:>6.1f} {2:>8.3f}'.format('Avg.', mean(cell_fps), weighted_average)
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
	parser.add_argument('--weighted-average','-w', type=int,   default=0,
		help='Whether or not traces to weight traces by their duration.')
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
			header = 'Trace'
			if args.weighted_average:
				header = 'Cell'
			if args.method.lower().startswith('c'):
				print '{0:>5} {1:>7} {2}'.format(header, 'FPS ', 'Correlation')
			elif args.method.lower().startswith('a'):
				print '{0:>5} {1:>7} {2}'.format(header, 'FPS ', 'AUC')
			else:
				print '{0:>5} {1:>7} {2}'.format(header, 'FPS ', 'Information gain')

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
				if args.weighted_average:
					print_weighted_average(R, data, ds)
				else:
					print_traces(R, fps)

		elif args.method.lower().startswith('a'):
			# compute correlations
			A = evaluate(data, method=args.method,
				optimize=args.optimize,
				downsampling=ds,
				verbosity=args.verbosity)

			auc.append(A)

			if args.verbosity > 0:
				if args.weighted_average:
					print_weighted_average(A, data, ds)
				else:
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
				if args.weighted_average:
					print_weighted_average(H + L, data, ds)
				else:
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
