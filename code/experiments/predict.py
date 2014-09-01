#!/usr/bin/env python

"""
This code needs to be rewritten.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from pickle import load
from numpy import corrcoef, mean
from tools import Experiment
from calcium import predict, preprocess

def main(argv):
	experiment = Experiment()

	# parse input arguments
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--models', '-m', type=str, required=True)
	parser.add_argument('--dataset', '-d', type=str, required=True)
	parser.add_argument('--preprocess', '-p', type=int, default=0)

	args = parser.parse_args(argv[1:])

	with open(args.dataset) as handle:
		data = load(handle)

	if args.preprocess:
		data = preprocess(data)

	results = Experiment(args.models)

	data = predict(data, results['models'], verbosity=1)
#
#	cc_all = []
#	fps_all = []
#
#	for cell_id, entry in enumerate(data):
#		predictions = entry['predictions'].ravel()
#		spikes = entry['spikes'].ravel()
#		fps = entry['fps']
#		fps_all.append([fps])
#		cc = corrcoef(predictions, spikes)[0, 1]
#		cc_all.append([cc])
#
#		print '{0:5}'.format(cell_id),
#		print '{0:5.3f} ({1:3.1f} Hz)'.format(cc, fps),
#
#		for _ in range(3):
#			# reduce sampling rate
#			if spikes.size % 2:
#				spikes = spikes[:-2:2] + spikes[1::2]
#				predictions = predictions[:-2:2] + predictions[1::2]
#			else:
#				spikes = spikes[::2] + spikes[1::2]
#				predictions = predictions[::2] + predictions[1::2]
#
#			fps /= 2.
#			fps_all[-1].append(fps)
#			cc = corrcoef(predictions, spikes)[0, 1]
#			cc_all[-1].append(cc)
#
#			print '{0:5.3f} ({1:3.1f} Hz)'.format(cc, fps),
#
#		print
#
#	
#	print '=' * 70
#	print '{0:5}'.format('Avg.'),
#
#	cc_all = mean(cc_all, 0)
#	fps_all = mean(fps_all, 0)
#
#	for cc, fps in zip(cc_all, fps_all):
#		print '{0:5.3f} ({1:3.1f} Hz)'.format(cc, fps),

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
