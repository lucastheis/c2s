#!/usr/bin/env python

"""
Measure the performance of NNP based spike prediction by repeatedly
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
from calcium import train, predict, preprocess, generate_inputs_and_outputs
from nnp import NNP
from tools import Experiment

def main(argv):
	experiment = Experiment()

	parser = ArgumentParser(argv[0], description=__doc__)

	parser.add_argument('--dataset',        '-d', type=str,   required=True)
	parser.add_argument('--num_hiddens',    '-n', type=int,   default=[3], nargs='+')
	parser.add_argument('--num_models',     '-m', type=int,   default=4)
	parser.add_argument('--var_explained',  '-e', type=float, default=95.)
	parser.add_argument('--window_length',  '-w', type=float, default=1000.)
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

		inputs, outputs, results = generate_inputs_and_outputs(
			data=[entry for j, entry in enumerate(data) if j != i],
			window_length=args.window_length,
			var_explained=args.var_explained)

		results['models'] = []

		for _ in range(args.num_models):
			nnp = NNP(inputs.shape[0], args.num_hiddens)
			nnp.train(inputs, outputs)

			results['models'].append(nnp)

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
