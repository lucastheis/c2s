#!/usr/bin/env python

"""
Remove trends and normalize sampling rate.

Examples:

	c2s preprocess data.pck data.preprocessed.pck
	c2s preprocess -t mat data.mat data.preprocessed.mat
"""

import sys

from argparse import ArgumentParser
from pickle import load, dump
from scipy.io import savemat
from c2s import preprocess, load_data

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('input',             type=str)
	parser.add_argument('output',            type=str, nargs='+')
	parser.add_argument('--filter',    '-s', type=int,   default=0)
	parser.add_argument('--fps',       '-f', type=float, default=100.,
		help='Up- or downsample data to match this sampling rate (100 fps).' )
	parser.add_argument('--verbosity', '-v', type=int,   default=1)

	args = parser.parse_args(argv[1:])

	# load data
	data = load_data(args.input)

	# preprocess data
	data = preprocess(
		data,
		fps=args.fps,
		filter=args.filter if args.filter > 0 else None,
		verbosity=args.verbosity)

	for filepath in args.output:
		if filepath.lower().endswith('.mat'):
			# store in MATLAB format
			savemat(output_file + '.mat', {'data': data})
		else:
			with open(output_file + '.pck', 'w') as handle:
				dump(data, handle, protocol=2)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
