#!/usr/bin/env python

"""
Summarize dataset.

Examples:

	c2s info data.pck
"""

import sys

from argparse import ArgumentParser
from pickle import dump
from scipy.io import savemat
from numpy import corrcoef, mean
from c2s import load_data

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('dataset',            type=str)

	args = parser.parse_args(argv[1:])

	# load data
	data = load_data(args.dataset)

	def prints(left, right):
		print('{0:<10} {1}'.format(left, right))

	prints('Average sampling rate:', mean([entry['fps'] for entry in data]))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
