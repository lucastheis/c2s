"""
Visualize nonlinearities optimized with respect to Poisson likelihood.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from pickle import load
from numpy import argmin, abs, mean, max, hstack, percentile
from scipy.io import loadmat
from tools import Experiment
from comparison import *
from calcium import downsample
from pgf import *

datasets = ['AOD', 'EUL']
dataset_labels = ['V1/OGB1', 'Retina/OGB1']
dataset_filepaths = ['data/data.3.preprocessed.pck', 'data/data.4.preprocessed.pck']
methods = ['STM', 'FOO', 'YAK', 'RAW']
method_labels = ['STM', 'Vogelstein et al. (2010)', 'Yaksi \& Friedrich (2006)', 'Raw']

parameters = {
	'STM': {
		'xmax': 2,
		'ymax': 2,
		'xtick': [0, 1, 2],
		'ytick': [0, 1, 2],
		'cutoff': 0,
		'num_bins': 300},
	'FOO': {
		'xmax': 2,
		'ymax': 2,
		'xtick': [0, 1, 2],
		'ytick': [0, 1, 2],
		'cutoff': 80,
		'num_bins': 80},
	'YAK': {
		'xmax': 2,
		'ymax': 2,
		'xtick': [0, 1, 2],
		'ytick': [0, 1, 2],
		'cutoff': 0,
		'num_bins': 80},
	'RAW': {
		'xmax': 2,
		'ymax': 2,
		'xtick': [0, 1, 2],
		'ytick': [0, 1, 2],
		'cutoff': 0,
		'num_bins': 100}}

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--fps', '-f', type=float, default=25.)

	args = parser.parse_args(argv[1:])

	figure(sans_serif=True)

	for i, dataset in enumerate(datasets):
		for j, method in enumerate(methods):
			# load nonlinearities
			results = Experiment(eval('lik_{0}_{1}'.format(dataset, method)))

			# load predictions
			if method == 'RAW':
				with open(dataset_filepaths[i]) as handle:
					data = load(handle)
					predictions = [entry['calcium'] for entry in data]
			else:
				if results['args'].results.endswith('.mat'):
					predictions = loadmat(results['args'].results)['predictions'].ravel()
				else:
					predictions = Experiment(results['args'].results)['predictions']

			# reduce sampling rate of predictions
			fps_max = max(mean(results['fps'], 1))
			predictions = hstack([downsample(_, fps_max / args.fps) for _ in predictions])
			
			idx = argmin(abs(mean(results['fps'], 1) - args.fps))
			f = results['f'][idx]

			if method == 'RAW':
				predictions = predictions / (fps_max / args.fps) / 1.5
				f = (f[0] / (fps_max / args.fps) / 1.5, f[1])

			subplot(i, j, spacing=1)
			F = predictions[predictions > percentile(predictions, parameters[method]['cutoff'])]
			h = hist(F, parameters[method]['num_bins'], density=True, color=RGB(.8, .8, .8))
			plot([0, parameters[method]['ymax']], [0, parameters[method]['ymax']], '---', color=RGB(.5, .5, .5))
			plot(f[0], f[1], '-', color=eval('color_{0}'.format(method)), line_width=3)
			xtick(parameters[method]['xtick'])
			ytick(parameters[method]['ytick'])

			if i == len(datasets) - 1:
				title(method_labels[j])
			if j == 0:
				ylabel(dataset_labels[i])

			h.const_plot = False
			axis(
				width=4,
				height=4,
				xmin=0,
				xmax=parameters[method]['xmax'],
				ymin=0,
				ymax=parameters[method]['ymax'])

	savefig('figures/nonlinearities.pdf')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
