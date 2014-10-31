#!/usr/bin/env python

"""
Trains STMs for the prediction of spikes from calcium traces.

Examples:

	c2s train -d data.pck
"""

import os
import sys

from argparse import ArgumentParser, HelpFormatter
from pickle import load
from numpy import mean, corrcoef
from numpy.random import rand, randint
from cmt.utils import random_select
from c2s import train, predict, preprocess
from c2s.experiment import Experiment

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__,
		formatter_class=lambda prog: HelpFormatter(prog, max_help_position=10, width=120))
	parser.add_argument('dataset',                type=str, nargs='+',
		help='Dataset(s) used for training.')
	parser.add_argument('--num_components', '-c', type=int,   default=3,
		help='Number of components used in STM model (default: %(default)d).')
	parser.add_argument('--num_features',   '-f', type=int,   default=2,
		help='Number of quadratic features used in STM model (default: %(default)d).')
	parser.add_argument('--num_models',     '-m', type=int,   default=4,
		help='Number of models trained (predictions will be averaged across models, default: %(default)d).')
	parser.add_argument('--keep_all',       '-k', type=int,   default=1,
		help='If set to 0, only the best model of all trained models is kept (default: %(default)d).')
	parser.add_argument('--finetune',       '-n', type=int,   default=0,
		help='If set to 1, enables another finetuning step which is performed after training (default: %(default)d).')
	parser.add_argument('--num_train',      '-t', type=int,   default=0,
		help='If specified, a (random) subset of cells is used for training.')
	parser.add_argument('--num_valid',      '-s', type=int,   default=0,
		help='If specified, a (random) subset of cells will be used for early stopping based on validation error.')
	parser.add_argument('--var_explained',  '-e', type=float, default=95.,
		help='Controls the degree of dimensionality reduction of fluorescence windows (default: %(default).0f).')
	parser.add_argument('--window_length',  '-w', type=float, default=1000.,
		help='Length of windows extracted from calcium signal for prediction (in milliseconds, default: %(default).0f).')
	parser.add_argument('--regularize',     '-r', type=float, default=0.,
		help='Amount of parameter regularization (filters are regularized for smoothness, default: %(default).1f).')
	parser.add_argument('--preprocess',     '-p', type=int,   default=0,
		help='If the data is not already preprocessed, this can be used to do it.')
	parser.add_argument('--output',         '-o', type=str,   default='results/',
		help='Directory or file where trained models will be stored.')
	parser.add_argument('--verbosity',      '-v', type=int,   default=1)

	args, _ = parser.parse_known_args(argv[1:])

	experiment = Experiment()

	if not args.datasets:
		print 'You have to specify at least 1 dataset.'
		return 0

	data = []
	for dataset in args.dataset:
		with open(dataset) as handle:
			data = data + load(handle)

	if args.preprocess:
		data = preprocess(data, args.verbosity)

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
		regularize=args.regularize,
		verbosity=args.verbosity)

	experiment['args'] = args
	experiment['training_cells'] = training_cells
	experiment['models'] = models

	if os.path.isdir(args.output) or args.output == 'results/':
		experiment.save(os.path.join(args.output, 'train.{0}.{1}.xpck'))
	else:
		experiment.save(args.output)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
