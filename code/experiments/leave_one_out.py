#!/usr/bin/env python

"""
Measure the performance of STM based spike prediction by repeatedly
using all but one cell for training and the remaining cell for testing.
"""

import os
import sys

sys.path.append('./code')

from argparse import ArgumentParser
from pickle import load
from numpy import mean, std, corrcoef, sqrt
from numpy.random import rand, randint
from cmt.utils import random_select
from calcium import train, predict, preprocess
from tools import Experiment

def main(argv):
	experiment = Experiment()

	parser = ArgumentParser(argv[0], description=__doc__)

	parser.add_argument('--dataset',        '-d', type=str,   required=True)
	parser.add_argument('--num_components', '-c', type=int,   default=3)
	parser.add_argument('--num_features',   '-f', type=int,   default=2)
	parser.add_argument('--num_models',     '-m', type=int,   default=4)
	parser.add_argument('--keep_all',       '-k', type=int,   default=1)
	parser.add_argument('--finetune',       '-n', type=int,   default=0)
	parser.add_argument('--num_valid',      '-v', type=int,   default=0)
	parser.add_argument('--var_explained',  '-e', type=float, default=95.)
	parser.add_argument('--window_length',  '-w', type=float, default=1000.)
	parser.add_argument('--regularize',     '-r', type=float, default=0.)
	parser.add_argument('--preprocess',     '-p', type=int,   default=0)
	parser.add_argument('--output',         '-o', type=str,   default='results/')

	args, _ = parser.parse_known_args(argv[1:])

	with open(args.dataset) as handle:
		data = load(handle)

	if args.preprocess:
		data = preprocess(data)

	predictions = []
	correlations = []

	for i in range(len(data)):
		print 'Test cell: {0}'.format(i)

		# train on all cells but cell i
		results = train(
			data=[entry for j, entry in enumerate(data) if j != i],
			num_valid=args.num_valid,
			num_models=args.num_models,
			var_explained=args.var_explained,
			window_length=args.window_length,
			keep_all=args.keep_all,
			finetune=args.finetune,
			model_parameters={
					'num_components': args.num_components,
					'num_features': args.num_features},
			training_parameters={
				'verbosity': 0},
			regularize=args.regularize,
			verbosity=1)

		# predict responses of cell i
		predictions.append(
			predict(data[i], results, verbosity=1)[0]['predictions'])

		# compute correlation with true spikes
		correlations.append(
			corrcoef(data[i]['spikes'], predictions[i])[0, 1])

		print 'Correlation: {0:.3f}'.format(correlations[-1])

	print 'Correlation: {0:.3f} +-  {1:.3f} (SEM)'.format(
			mean(correlations),
			std(correlations) / sqrt(len(correlations)))

	experiment['args'] = args
	experiment['correlations'] = correlations
	experiment['predictions'] = predictions

	if os.path.isdir(args.output):
		experiment.save(os.path.join(args.output, 'leave_one_out.{0}.{1}.xpck'))
	else:
		experiment.save(args.output)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
