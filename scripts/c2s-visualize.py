#!/usr/bin/env python

"""
Plot calcium traces and spike trains.

Examples:

	c2s visualize data.pck
"""

import sys

from argparse import ArgumentParser
from pickle import dump
from scipy.io import savemat
from numpy import corrcoef, mean, arange, logical_and
from c2s import load_data, preprocess

try:
	import matplotlib
	matplotlib.use('Agg')
	from matplotlib import pyplot as plt
except ImportError:
	print 'Install `matplotlib` first.'
	sys.exit(1)

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('dataset',            type=str)
	parser.add_argument('--preprocess', '-p', type=int, default=0)
	parser.add_argument('--output',     '-o', type=str, default='')
	parser.add_argument('--seconds',    '-S', type=int, default=60)
	parser.add_argument('--offset',     '-O', type=int, default=0)
	parser.add_argument('--width',      '-W', type=int, default=10)
	parser.add_argument('--height',     '-H', type=int, default=0)
	parser.add_argument('--cells',      '-c', type=int, default=[], nargs='+')
	parser.add_argument('--dpi',        '-D', type=int, default=100)
	parser.add_argument('--font',       '-F', type=str, default='Arial')

	args = parser.parse_args(argv[1:])

	# load data
	data = load_data(args.dataset)
	cells = args.cells if args.cells else range(1, len(data) + 1)
	data = [data[c - 1] for c in cells]

	if args.preprocess:
		data = preprocess(data)

	plt.rcParams['font.family'] = args.font
	plt.rcParams['savefig.dpi'] = args.dpi

	plt.figure(figsize=(
		args.width,
		args.height if args.height > 0 else len(data) * 1.5 + .3))

	for k, entry in enumerate(data):
		offset = int(entry['fps'] * args.offset)
		length = int(entry['fps'] * args.seconds)
		calcium = entry['calcium'].ravel()[offset:offset + length]

		plt.subplot(len(data), 1, k + 1)
		plt.plot(args.offset + arange(calcium.size) / entry['fps'], calcium,
			color=(.1, .6, .4))

		if 'spike_times' in entry:
			spike_times = entry['spike_times'].ravel() / 1000.
			spike_times = spike_times[logical_and(
				spike_times > args.offset,
				spike_times < args.offset + args.seconds)]

			for st in spike_times:
				plt.plot([st, st], [-1, -.5], 'k', lw=1.5)

		plt.yticks([])
		plt.ylim([-2., 5.])
		plt.xlim([args.offset, args.offset + args.seconds])
		plt.ylabel('Cell {0}'.format(cells[k]))
		plt.grid()

		if k < len(data) - 1:
			plt.xticks(plt.xticks()[0], [])

	plt.xlabel('Time [seconds]')
	plt.tight_layout()

	if args.output:
		plt.savefig(args.output)
	else:
		plt.show()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
