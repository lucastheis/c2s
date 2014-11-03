#!/usr/bin/env python

"""
Predict firing rates from calcium traces.

Examples:

	c2s predict -p 1 data.pck predictions.xpck
	c2s predict -m model.xpck data.preprocessed.pck predictions.xpck
	c2s predict -t mat data.preprocessed.pck predictions.mat
"""

import sys

from argparse import ArgumentParser
from pickle import dump
from scipy.io import savemat
from numpy import corrcoef, mean
from c2s import predict, preprocess, load_data
from c2s.experiment import Experiment

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('dataset',            type=str)
	parser.add_argument('output',             type=str)
	parser.add_argument('--models',     '-m', type=str, default='')
	parser.add_argument('--preprocess', '-p', type=int, default=0,
		help='If you haven\'t already applied `preprocess` to the data, set to 1 (default: 0).')
	parser.add_argument('--verbosity',  '-v', type=int, default=1)
	parser.add_argument('--type',       '-t', type=str, choices=['xpck', 'pck', 'mat', 'both'], default='xpck',
		help='Whether to save output in x-pickle format, MATLAB format, or both.')

	args = parser.parse_args(argv[1:])

	experiment = Experiment()

	# load data
	data = load_data(args.dataset)

	if args.preprocess:
		# preprocess data
		data = preprocess(data, args.verbosity)

	if args.models:
		# load training results
		results = Experiment(args.models)['models']
	else:
		# use default model
		results = None

	# predict firing rates
	data = predict(data, results, verbosity=args.verbosity)

	# remove data besides predictions
	for entry in data:
		del entry['calcium']
		if 'spikes' in entry:
			del entry['spikes']
		if 'spike_times' in entry:
			del entry['spike_times']

	output_file = args.output

	if output_file.endswith('.xpck') or output_file.endswith('.mat') or output_file.endswith('.pck'):
		# remove file ending
		output_file = '.'.join(output_file.split('.')[:-1])

	if args.type in ['xpck', 'both']:
		with open(output_file + '.xpck', 'w') as handle:
			dump(data, handle)

	if args.type in ['mat', 'both']:
		# store in MATLAB format
		savemat(output_file + '.mat', {'data': data})

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
