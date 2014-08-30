"""
Visualize calcium traces and predictions ordered by quality of predictions (performance).
"""

import os
import sys

sys.path.append('./code')

from argparse import ArgumentParser
from numpy import linspace, asarray, hstack, zeros_like, logical_and, convolve, min, mean
from numpy import argmin, argsort, ceil, arange, max
from scipy.signal import hamming
from scipy.io import loadmat
from pickle import load
from pgf import *
from tools import Experiment
from calcium import downsample
from comparison import *

# choose which datasets to visualize
#datasets = ['AOD', 'EUL']
#dataset_paths = ['data/data.3.preprocessed.pck', 'data/data.4.preprocessed.pck']
#datasets = ['AOD']
#dataset_paths = ['data/data.3.preprocessed.pck']
#datasets = ['GC6']
#dataset_paths = ['data/data.2.preprocessed.pck']
#datasets = ['EUL']
#dataset_paths = ['data/data.4.preprocessed.pck']
#datasets = ['GLV']
#dataset_paths = ['data/data.6.preprocessed.pck']
datasets = ['GAR']
dataset_paths = ['data/data.7.preprocessed.pck']

# choose which methods to visualize
methods = ['STM', 'STX', 'FOO', 'YAK']

def df_over_f(signal, fps, tau=.5):
	f = hamming(int(fps * 2. / tau + 1.5))
	f = f / sum(f)
	signal = signal - min(signal) + 1e-16
	return signal / convolve(signal, f, 'same')

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--seconds',     '-s', type=int, default=60)
	parser.add_argument('--offset',      '-o', type=int, default=50)
	parser.add_argument('--downsample',  '-d', type=int, default=4)
	parser.add_argument('--fps',         '-f', type=int, default=25)
	parser.add_argument('--max_traces',  '-m', type=int, default=1)
	parser.add_argument('--width',       '-w', type=int, default=6)
	parser.add_argument('--normalize',   '-n', type=int, default=1)
	parser.add_argument('--sort_by',     '-b', type=str, default='correlations')

	args = parser.parse_args(argv[1:])

	for l, dataset in enumerate(datasets):
		pdfs = []
		performance = {}
		predictions = {}

		# load predictions and performance measures
		for method in methods:
			if method == 'RAW':
				continue

			# load performance measures
			if args.sort_by.lower().startswith('c'):
				results = Experiment(eval('corr_{0}_{1}'.format(dataset, method)))
				idx = argmin(abs(mean(results['fps'], 1) - args.fps))
				performance[method] = results['correlations'][idx]

			elif args.sort_by.lower().startswith('a'):
				results = Experiment(eval('auc_{0}_{1}'.format(dataset, method)))
				idx = argmin(abs(mean(results['fps'], 1) - args.fps))
				performance[method] = results['auc'][idx]

			elif args.sort_by.lower().startswith('l'):
				results = Experiment(eval('loglik_{0}_{1}'.format(dataset, method)))
				idx = argmin(abs(mean(results['fps'], 1) - args.fps))
				performance[method] = results['loglik'][idx]

			else:
				print 'Unknown criterion \'{0}\'.'.format(args.sort_by)
				return 0

			if not os.path.exists(results['args'].results):
				print 'Could not find "{0}".'.format(results['args'].results)
				return 0

			print 'Loading {0}...'.format(results['args'].results)

			# load predictions
			if results['args'].results.endswith('.mat'):
				predictions[method] = loadmat(results['args'].results)['predictions'].ravel()
			else:
				results = Experiment(results['args'].results)
				predictions[method] = results['predictions']

		with open(dataset_paths[l]) as handle:
			data = load(handle)

		# sort predictions by performance of first method
		indices = argsort(performance[methods[0]])

		# split dataset in chunks and show in separate plots
		for b in range(int(ceil(len(data) / float(args.max_traces)) + .5)):
			figure(sans_serif=True)

			print 'Creating figure for dataset {0}, batch {1}...'.format(dataset, b)

			# for each trial in this chunk
			for j, i in enumerate(indices[b * args.max_traces:(b + 1) * args.max_traces]):
				subplot(j, 0)

				spikes = downsample(data[i]['spikes'].ravel(), args.downsample)
				calcium = downsample(data[i]['calcium'].ravel(), args.downsample)

				max_spikes = max(spikes) / 3.
				num_spikes = sum(spikes)

				spikes = spikes / max_spikes

				fps = data[i]['fps'] / args.downsample
				bins = fps * args.seconds
				offset = fps * args.offset

				spike_times = data[i]['spike_times'].ravel() / 1000. * fps - offset
				spike_times = spike_times[logical_and(spike_times > 0, spike_times < bins)]

				# plot calcium trace
				plot(calcium[offset:offset + bins] / args.downsample, 'k', line_width=.25)

				# plot firing rate
				plot(arange(bins) - .5, spikes[offset:offset + bins] / args.downsample * 2.5 - 2.5, '-',
					color=RGB(0.5, 0.5, 0.5), line_width=.25, const_plot=True)

				for k, method in enumerate(methods):
					pred = downsample(predictions[method][i].ravel(), args.downsample)
					if args.normalize:
						# normalize by integral
						pred = pred / sum(pred) * num_spikes / max_spikes
					else:
						# normalize by maximum
						pred = pred / (max(pred[offset:offset + bins]) + .1) * 2.5

					plot(arange(bins) - .5, pred[offset:offset + bins] / args.downsample * 1.5 - 3.5 - k, '-',
						color=eval('color_{0}'.format(method)),
						line_width=.25, const_plot=True)

				# plot spikes
				for st in spike_times:
					plot([st, st], [-.75, -.5], 'k', line_width=.5)

				num_ticks = int(args.seconds / 10. + .5) + 1
				xtick(linspace(0, bins, num_ticks), asarray(linspace(0, bins, num_ticks) / fps + .5, dtype='int'))
				ytick([])
				ylim(-3 - len(methods), 4.5)
				xlabel('Time [s]')
				axis(width=args.width, height=1.8295 + .261 * len(methods))

				corr_str = []
				for m in methods:
					color = eval('color_{0}'.format(m))
					corr_str.append('{{\\color[rgb]{{{0:.2f}, {1:.2f}, {2:.2f}}} {3:.4f}}}'.format(
						color.red / 255., color.green / 255., color.blue / 255., performance[m][i]))
				corr_str = ' / '.join(corr_str)

				title('\\small {2}: {0} (at {1} fps)'.format(corr_str, args.fps, args.sort_by))
				box('off')
				axis('off')

			plot([12 * fps, 22 * fps], [3.1, 3.1], 'k-', line_width=2)
			text(17 * fps, 4.0, '\\small 10 sec')

			savefig('figures/predictions.{0}.{1}fps.{2}.tex'.format(dataset, args.fps, b))
			try:
				filename = 'figures/predictions.{0}.{1}fps.{2}.pdf'.format(dataset, args.fps, b)
				savefig(filename)
				pdfs.append(filename)
			except RuntimeError:
				print 'Compilation seems to have failed...'

#		try:
		os.system('pdftk {0} cat output {1}'.format(
			' '.join(pdfs[::-1]),
			'figures/predictions.{0}.{1}fps.pdf'.format(dataset, args.fps)))
#		except:
#			pass

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
