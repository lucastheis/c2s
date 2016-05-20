#!/usr/bin/env python

"""
Measure the performance of STM based spike prediction by repeatedly
using all but one cell for training and the remaining cell for testing.
"""

import os
import sys

from argparse import ArgumentParser
from pickle import dump
from scipy.io import savemat
from numpy import mean, std, corrcoef, sqrt, unique
from numpy.random import rand, randint
from cmt.utils import random_select
from c2s import load_data, train, predict, preprocess
from c2s.experiment import Experiment
from c2s.utils import convert

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('dataset',                type=str,   nargs='+')
	parser.add_argument('output',                 type=str)
	parser.add_argument('--num_components', '-c', type=int,   default=3)
	parser.add_argument('--num_features',   '-f', type=int,   default=2)
	parser.add_argument('--num_models',     '-m', type=int,   default=4)
	parser.add_argument('--keep_all',       '-k', type=int,   default=1)
	parser.add_argument('--finetune',       '-n', type=int,   default=0)
	parser.add_argument('--num_valid',      '-s', type=int,   default=0)
	parser.add_argument('--var_explained',  '-e', type=float, default=95.)
	parser.add_argument('--window_length',  '-w', type=float, default=1000.)
	parser.add_argument('--regularize',     '-r', type=float, default=0.)
	parser.add_argument('--preprocess',     '-p', type=int,   default=0)
	parser.add_argument('--verbosity',      '-v', type=int,   default=1)

	args, _ = parser.parse_known_args(argv[1:])

	experiment = Experiment()

	# load data
	data = []
	for dataset in args.dataset:
		data = data + load_data(dataset)

	# preprocess data
	if args.preprocess:
		data = preprocess(data)

	# list of all cells
	if 'cell_num' in data[0]:
		# several trials/entries may belong to the same cell
		cells = unique([entry['cell_num'] for entry in data])
	else:
		# one cell corresponds to one trial/entry
		cells = range(len(data))
		for i in cells:
			data[i]['cell_num'] = i

	for i in cells:
		data_train = [entry for entry in data if entry['cell_num'] != i]
		data_test = [entry for entry in data if entry['cell_num'] == i]

		if args.verbosity > 0:
			print 'Test cell: {0}'.format(i)

		# train on all cells but cell i
		results = train(
			data=data_train,
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

		if args.verbosity > 0:
			print 'Predicting...'

		# predict responses of cell i
		predictions = predict(data_test, results, verbosity=0)

		for entry1, entry2 in zip(data_test, predictions):
			entry1['predictions'] = entry2['predictions']

	# remove data except predictions
	for entry in data:
		if 'spikes' in entry:
			del entry['spikes']
		if 'spike_times' in entry:
			del entry['spike_times']
		del entry['calcium']

	# save results
	if args.output.lower().endswith('.mat'):
		savemat(args.output, convert({'data': data}))

	elif args.output.lower().endswith('.xpck'):
		experiment['args'] = args
		experiment['data'] = data
		experiment.save(args.output)

	else:
		with open(args.output, 'w') as handle:
			dump(data, handle, protocol=2)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
