#!/usr/bin/env python

"""
This code needs to be rewritten.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from pickle import load
from numpy import corrcoef, mean
from calcium import predict, preprocess
from calcium.experiment import Experiment

def main(argv):
	experiment = Experiment()

	# parse input arguments
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--dataset',    '-d', type=str, required=True)
	parser.add_argument('--models',     '-m', type=str, default='')
	parser.add_argument('--preprocess', '-p', type=int, default=0)
	parser.add_argument('--verbosity',  '-v', type=int, default=1)

	args = parser.parse_args(argv[1:])

	# load data
	with open(args.dataset) as handle:
		data = load(handle)

	if args.preprocess:
		# preprocess data
		data = preprocess(data, args.verbsoity)

	if args.models:
		# load training results
		results = Experiment(args.models)['models']
	else:
		# use default model
		results = None

	# predict firing rates
	data = predict(data, results, verbosity=args.verbosity)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
