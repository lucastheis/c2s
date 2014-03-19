"""
Plots performance as a function of the sampling rate.
"""

import os
import sys

sys.path.append('./code')

from numpy import array, hstack
from pgf import *
from pgf.colormap import colormaps
from tools import Experiment
from comparison import *

filepath = 'figures/performance_curves.pdf'

#datasets = ['EUL', 'AOD']
#methods = ['STM', 'STX', 'FOO', 'SOO', 'YAK', 'RAW']
#method_labels = ['STM', 'STM$^*$', 'FAST-OOPSI', 'SMC-OOPSI', 'DECONV', 'RAW']

datasets = ['AOD']
methods = ['STM', 'NNP', 'FOO', 'LNP']
method_labels = ['STM', 'NNP', 'FAST-OOPSI', 'LNP']

def get_corr(filepath):
	"""
	Extracts average correlations from experiment.
	"""

	if not os.path.exists(filepath):
		print filepath, 'does not exist.'
		return 0., 0.

	results = Experiment(filepath)

	N = asarray(results['correlations']).shape[1]

	fps = mean(results['fps'], 1)
	corr = mean(results['correlations'], 1)
	sem = std(results['correlations'], 1, ddof=1) / sqrt(N)

	return fps, corr, sem



def get_info(filepath):
	"""
	Extracts average information rates from experiment.
	"""

	if not os.path.exists(filepath):
		print filepath, 'does not exist.'
		return 0., 0.

	results = Experiment(filepath)

	fps = mean(results['fps'], 1)
	info = asarray(results['entropy']) + asarray(results['loglik'])
	sem = std(info, 1, ddof=1) / sqrt(info.shape[1])
	info = mean(info, 1)

	return fps, info, sem



def main(argv):
	figure(sans_serif=True, margin=4.)

	# compute Loftus & Masson's standard error
	sem_adjusted = []

	for dataset in datasets:
		sem_adjusted.append([])

		corr = []
		for method in methods:
			corr.append(Experiment(eval('corr_{0}_{1}'.format(dataset, method)))['correlations'])
		corr = array(corr)

		for n in range(corr.shape[1]):
			# compute Loftus & Masson standard error for n-th sampling rate
			sem_adjusted[-1].append(sem_lm(corr[:, n, :]))

	sem_adjusted = array(sem_adjusted)

	# plot correlations
	for k, dataset in enumerate(datasets):
		subplot(k, 0, spacing=3)

		for method in methods:
			fps, corr, sem = get_corr(eval('corr_{0}_{1}'.format(dataset, method)))

			plot(
				hstack([fps, fps[::-1]]),
				hstack([corr + 2. * sem_adjusted[k], corr[::-1] - 2. * sem_adjusted[k][::-1]]),
				fill=eval('color_{0}'.format(method)),
				opacity=.1,
				pgf_options=['forget plot', 'draw=none'])
			plot(fps, corr, '-',
				color=eval('color_{0}'.format(method)),
				line_width=2.)


		xlabel('Sampling rate [Hz]')
		ylabel(r'Correlation $\pm$ 2 $\cdot$ SEM$^\text{L\&M}$')
		box('off')
		axis(width=5, height=5, xmin=0., ymin=0., ymax=1.)
		title(r'\textbf{' + dataset + '}')

	# compute Loftus & Masson's standard error
	sem_adjusted = []

	for dataset in datasets:
		sem_adjusted.append([])

		info = []
		for method in methods:
			results = Experiment(eval('lik_{0}_{1}'.format(dataset, method)))
			info.append(array(results['entropy']) + array(results['loglik']))
		info = array(info)

		for n in range(info.shape[1]):
			# compute Loftus & Masson standard error for n-th sampling rate
			sem_adjusted[-1].append(sem_lm(info[:, n, :]))

	sem_adjusted = array(sem_adjusted)

	# plot information rates
	for k, dataset in enumerate(datasets):
		subplot(k, 1)

		for method in methods:
			fps, info, sem = get_info(eval('lik_{0}_{1}'.format(dataset, method)))

			plot(
				hstack([fps, fps[::-1]]),
				hstack([info + 2. * sem_adjusted[k], info[::-1] - 2. * sem_adjusted[k][::-1]]),
				fill=eval('color_{0}'.format(method)),
				opacity=.1,
				pgf_options=['forget plot', 'draw=none'])
			plot(fps, info, '-',
				color=eval('color_{0}'.format(method)),
				line_width=2.)
		xlabel('Sampling rate [Hz]')
		ylabel(r'Information gain $\pm$ 2 $\cdot$ SEM$^\text{L\&M}$ [bit/s]')
		box('off')
		axis(width=5, height=5, xmin=0., ymin=0., ymax=6.)
		title(r'\textbf{' + dataset + '}')

	legend(*method_labels, location='outer north east')

	savefig(filepath)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
