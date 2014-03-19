#!/usr/bin/env python

"""
Trains STMs on one dataset and predict responses of a second dataset.
"""

import os
import sys

sys.path.append('./code')

from argparse import ArgumentParser
from pickle import load
from numpy import mean, corrcoef
from numpy.random import rand, randint
from cmt.utils import random_select
from calcium import train, predict, preprocess
from tools import Experiment

def main(argv):
	experiment = Experiment()

	parser = ArgumentParser(argv[0], description=__doc__)

	parser.add_argument('--dataset_train',  '-r', type=str,   required=True)
	parser.add_argument('--dataset_test',   '-s', type=str,   required=True)
	parser.add_argument('--num_components', '-c', type=int,   default=3)
	parser.add_argument('--num_features',   '-f', type=int,   default=2)
	parser.add_argument('--num_models',     '-m', type=int,   default=4)
	parser.add_argument('--keep_all',       '-k', type=int,   default=1)
	parser.add_argument('--finetune',       '-n', type=int,   default=0)
	parser.add_argument('--num_train',      '-t', type=int,   default=-1)
	parser.add_argument('--num_valid',      '-v', type=int,   default=0)
	parser.add_argument('--var_explained',  '-e', type=float, default=95.)
	parser.add_argument('--window_length',  '-w', type=float, default=1000.)
	parser.add_argument('--preprocess',     '-p', type=int,   default=0)
	parser.add_argument('--output',         '-o', type=str,   default='results/')

	args, _ = parser.parse_known_args(argv[1:])

	# load training data
	with open(args.dataset_train) as handle:
		data = load(handle)

	if args.preprocess:
		data = preprocess(data)



	### TRAINING

	# pick cells for training
	if args.num_train > 0:
		training_cells = random_select(args.num_train, len(data))
	else:
		training_cells = range(len(data))

	models = train([data[cell_id] for cell_id in training_cells],
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
			'verbosity': 1},
		verbosity=1)



	### PREDICTION

	# load test data
	with open(args.dataset_test) as handle:
		data = load(handle)

	if args.preprocess:
		data = preprocess(data)

	data = predict(data, models, verbosity=1)



	### EVALUATION

	print 'Evaluating...'

	corr = []
	predictions = []

	for entry in data:
		predictions.append(entry['predictions'])
		corr.append(corrcoef(entry['predictions'], entry['spikes'])[0, 1])

	corr = mean(corr)

	print 'Correlation: {0:.5f}'.format(corr)

	experiment['args'] = args
	experiment['training_cells'] = training_cells
	experiment['models'] = models
	experiment['corr'] = corr
	experiment['predictions'] = predictions

	if os.path.isdir(args.output):
		experiment.save(os.path.join(args.output, 'generalize.{0}.{1}.xpck'))
	else:
		experiment.save(args.output)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
