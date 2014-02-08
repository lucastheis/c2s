#!/usr/bin/env python

"""
Trains STMs for the prediction of spikes from calcium traces.
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

	parser.add_argument('--dataset',        '-d', type=str,   required=True)
	parser.add_argument('--num_components', '-c', type=int,   default=randint(1, 5))
	parser.add_argument('--num_features',   '-f', type=int,   default=randint(0, 5))
	parser.add_argument('--num_models',     '-m', type=int,   default=randint(1, 5))
	parser.add_argument('--keep_all',       '-k', type=int,   default=randint(0, 2))
	parser.add_argument('--finetune',       '-n', type=int,   default=randint(0, 2))
	parser.add_argument('--num_train',      '-t', type=int,   default=randint(6, 15))
	parser.add_argument('--num_valid',      '-v', type=int,   default=randint(0, 5))
	parser.add_argument('--var_explained',  '-e', type=float, default=rand() * 20. + 80)
	parser.add_argument('--window_length',  '-w', type=float, default=rand() * 1900. + 200.)
	parser.add_argument('--preprocess',     '-p', type=int,   default=0)
	parser.add_argument('--output_dir',     '-o', type=str,   default='results/')

	args, _ = parser.parse_known_args(argv[1:])

	with open(args.dataset) as handle:
		data = load(handle)

	if args.preprocess:
		data = preprocess(data)



	### TRAINING

	# pick cells for training
	training_cells = random_select(args.num_train, len(data))

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

	data = predict(data, models, verbosity=1)



	### EVALUATION

	print 'Evaluating...'

	corr = []
	corr_train = []
	corr_test = []
	predictions = []

	for cell_id, entry in enumerate(data):
		predictions.append(entry['predictions'])
		corr.append(corrcoef(entry['predictions'], entry['spikes'])[0, 1])

		if cell_id in training_cells:
			corr_train.append(corr[-1])
		else:
			corr_test.append(corr[-1])

	corr = mean(corr)
	corr_test = mean(corr_test)
	corr_train = mean(corr_train)

	print 'Correlation:'
	print '\t{0:.5f} (training)'.format(corr_train)
	print '\t{0:.5f} (test)'.format(corr_test)
	print '\t{0:.5f} (total)'.format(corr)

	experiment['args'] = args
	experiment['training_cells'] = training_cells
	experiment['models'] = models
	experiment['corr'] = corr
	experiment['corr_test'] = corr_test
	experiment['corr_train'] = corr_train
	experiment['predictions'] = predictions

	experiment.save(os.path.join(args.output_dir, 'predictions.{0}.{1}.xpck'))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
