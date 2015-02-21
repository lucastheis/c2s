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
from numpy import corrcoef, mean, unique
from c2s import load_data

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('dataset',            type=str)

	args = parser.parse_args(argv[1:])

	# load data
	data = load_data(args.dataset)

	def prints(left, right):
		print('{0:<30} {1}'.format(left, right))

	num_spikes = 0
	length = 0
	for entry in data:
		length += entry['calcium'].size / float(entry['fps']) # seconds
		if 'spike_times' in entry:
			num_spikes += entry['spike_times'].size
		elif 'spikes' in entry:
			num_spikes += entry['spikes'].sum()

	if 'cell_num' in data[0]:
		num_cells = len(unique([entry['cell_num'] for entry in data]))
	else:
		num_cells = len(data)

	prints('Number of cells:', '{0}'.format(num_cells))
	prints('Number of traces:', '{0}'.format(len(data)))
	prints('Total length:', '{0} minutes, {1} seconds'.format(int(length) // 60, int(length) % 60))
	prints('Total number of spikes:', num_spikes)
	prints('Average firing rate:', '{0:.2f} [spike/sec]'.format(num_spikes / length))
	prints('Average sampling rate:', '{0:.1f}'.format(mean([entry['fps'] for entry in data])))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
