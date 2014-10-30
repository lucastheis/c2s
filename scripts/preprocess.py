#!/usr/bin/env python

"""
Preprocess data.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from pickle import load, dump
from scipy.io import savemat
from calcium import preprocess

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--input',     '-i', type=str,   required=True)
	parser.add_argument('--output',    '-o', type=str,   required=True)
	parser.add_argument('--filter',    '-s', type=int,   default=0)
	parser.add_argument('--fps',       '-f', type=float, default=100.)
	parser.add_argument('--matlab',    '-m', type=str,   default='')
	parser.add_argument('--verbosity', '-v', type=int,   default=1)

	args = parser.parse_args(argv[1:])

	with open(args.input) as handle:
		data = load(handle)

	data = preprocess(
		data,
		fps=args.fps,
		filter=args.filter if args.filter > 0 else None,
		verbosity=args.verbosity)

	with open(args.output, 'w') as handle:
		dump(data, handle)

	if args.matlab:
		# store in MATLAB format
		savemat(args.matlab, {'data': data})

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
