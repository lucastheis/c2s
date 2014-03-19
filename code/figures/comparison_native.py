"""
Generates bar plots of correlations and information rates.
"""

import os
import sys

sys.path.append('./code')

from numpy import argmax, abs, mean, std, sqrt, square, asarray, zeros, ones, triu
from pgf import *
from pgf.colormap import colormaps
from tools import Experiment

filepath = 'figures/comparison_native.pdf'

# colors corresponding to the different methods
cmap = 'jet'
color_STM = RGB(0, 0, 0)
color_STX = RGB(*colormaps[cmap].colors[0])
color_FOO = RGB(*colormaps[cmap].colors[80])
color_SOO = RGB(*colormaps[cmap].colors[120])
color_RAW = RGB(*colormaps[cmap].colors[180])
color_YAK = RGB(*colormaps[cmap].colors[220])

# locations of precomputed correlations
corr_AOD_STM = 'results/native/correlations.3.stm.xpck'
corr_AOD_STX = 'results/native/correlations.3.stmx.xpck'
corr_AOD_FOO = 'results/native/correlations.3.fast_oopsi.xpck'
corr_AOD_SOO = 'results/native/correlations.3.smc_oopsi.xpck'
corr_AOD_YAK = 'results/native/correlations.3.yaksi.xpck'
corr_AOD_RAW = 'results/native/correlations.3.raw.xpck'

corr_EUL_STM = 'results/native/correlations.4.stm.xpck'
corr_EUL_STX = 'results/native/correlations.4.stmx.xpck'
corr_EUL_FOO = 'results/native/correlations.4.fast_oopsi.xpck'
corr_EUL_SOO = 'results/native/correlations.4.smc_oopsi.xpck'
corr_EUL_YAK = 'results/native/correlations.4.yaksi.xpck'
corr_EUL_RAW = 'results/native/correlations.4.raw.xpck'

# locations of precomputed likelihoods
lik_AOD_STM = 'results/native/likelihoods.3.stm.xpck'
lik_AOD_STX = 'results/native/likelihoods.3.stmx.xpck'
lik_AOD_FOO = 'results/native/likelihoods.3.fast_oopsi.xpck'
lik_AOD_SOO = 'results/native/likelihoods.3.smc_oopsi.xpck'
lik_AOD_YAK = 'results/native/likelihoods.3.yaksi.xpck'
lik_AOD_RAW = 'results/native/likelihoods.3.raw.xpck'

lik_EUL_STM = 'results/native/likelihoods.4.stm.xpck'
lik_EUL_STX = 'results/native/likelihoods.4.stmx.xpck'
lik_EUL_FOO = 'results/native/likelihoods.4.fast_oopsi.xpck'
lik_EUL_SOO = 'results/native/likelihoods.4.smc_oopsi.xpck'
lik_EUL_YAK = 'results/native/likelihoods.4.yaksi.xpck'
lik_EUL_RAW = 'results/native/likelihoods.4.raw.xpck'

datasets = ['AOD', 'EUL']
methods = ['STM', 'FOO', 'SOO', 'YAK', 'RAW']
method_labels = ['STM', 'FAST-OOPSI', 'SMC-OOPSI', 'DECONV', 'RAW']



def get_corr(filepath):
	"""
	Extracts average correlation at given sampling rate from experiment.
	"""

	if not os.path.exists(filepath):
		print filepath, 'does not exist.'
		return 0., 0.

	results = Experiment(filepath)

	idx = argmax(abs(mean(results['fps'], 1)))

	N = asarray(results['correlations']).shape[1]

	corr = mean(results['correlations'], 1)[idx]
	sem = std(results['correlations'], 1, ddof=1)[idx] / sqrt(N)

	return corr, sem



def get_corr_all(filepath):
	"""
	Extracts all correlation at given sampling rate from experiment.
	"""

	results = Experiment(filepath)
	idx = argmax(abs(mean(results['fps'], 1)))
	return results['correlations'][idx]



def get_info(filepath):
	"""
	Extracts average information rate at given sampling rate from experiment.
	"""

	if not os.path.exists(filepath):
		print filepath, 'does not exist.'
		return 0., 0.

	results = Experiment(filepath)

	idx = argmax(abs(mean(results['fps'], 1)))

	info = asarray(results['entropy']) + asarray(results['loglik'])
	sem = std(info, 1, ddof=1)[idx] / sqrt(info.shape[1])
	info = mean(info, 1)[idx]

	return info, sem



def get_info_all(filepath):
	"""
	Extracts all information rates at given sampling rate from experiment.
	"""

	results = Experiment(filepath)
	idx = argmax(abs(mean(results['fps'], 1)))
	return asarray(results['entropy'][idx]) + asarray(results['loglik'][idx])



def sem_lm(values):
	"""
	Compute Loftus & Masson's (1994) standard error.

	@type  values: ndarray
	@param values: NxM array where N is the number of methods and M is the number of cells

	@rtype: float
	@return: Loftus & Masson standard error

	B{References:}

	- G. R. Loftus and M. E. J. Masson, Using confidence intervals in within-subject designs, 1994
	- V. H. Franz and G. R. Loftus, Standard errors and confidence intervals in within-subjects designs, 2012
	"""

	values = asarray(values)

	M = values.shape[0]
	N = values.shape[1]

	sem_diff = zeros([M, M])

	# compute standard errors of differences between methods
	for i in range(M):
		for j in range(i + 1, M):
			# differences in performance between methods i and j
			diffs = values[i] - values[j]
			sem_diff[i, j] = std(diffs, ddof=1) / sqrt(N)

	mask = triu(ones([M, M]), 1) > .5

	return sqrt(mean(square(sem_diff[mask]) / 2.))



def main(argv):
	figure(sans_serif=True, margin=4.)

	xval = range(1, len(datasets) + 1)

	# PLOT CORRELATIONS

	subplot(0, 0)

	# compute Loftus & Masson's standard error
	sem_adjusted = []

	for dataset in datasets:
		corr = []
		for method in methods:
			corr.append(get_corr_all(eval('corr_{0}_{1}'.format(dataset, method))))
		sem_adjusted.append(sem_lm(corr))

	# compute and plot average correlation
	for method in methods:
		yval = []
		yerr = []

		for k, dataset in enumerate(datasets):
			corr, sem = get_corr(eval('corr_{0}_{1}'.format(dataset, method)))
			yval.append(corr)
			yerr.append(sem_adjusted[k] * 2.)

		bar(xval, yval, yerr=yerr, color=eval('color_{0}'.format(method)), bar_width=.2)

	xtick(xval, datasets)
	axis(
		xmin=0.5,
		xmax=len(datasets) + .5,
		ymin=0.,
		width=2.5 * len(datasets),
		height=5)
	box('off')
	ylabel(r'Correlation $\pm$ 2 $\cdot$ SEM$^\text{L\&M}$')

	# PLOT INFORMATION RATES

	subplot(0, 1)

	# compute Loftus & Masson's standard error
	sem_adjusted = []

	for dataset in datasets:
		corr = []
		for method in methods:
			corr.append(get_info_all(eval('lik_{0}_{1}'.format(dataset, method))))
		sem_adjusted.append(sem_lm(corr))

	# compute and plot average information rate
	for method in methods:
		yval = []
		yerr = []

		for k, dataset in enumerate(datasets):
			corr, sem = get_info(eval('lik_{0}_{1}'.format(dataset, method)))
			yval.append(corr)
			yerr.append(sem_adjusted[k] * 2.)

		bar(xval, yval, yerr=yerr, color=eval('color_{0}'.format(method)), bar_width=.2)

	xtick(xval, datasets)
	axis(
		xmin=0.5,
		xmax=len(datasets) + .5,
		ymin=0.,
		width=2.5 * len(datasets),
		height=5)
	box('off')
	ylabel(r'Information gain $\pm$ 2 $\cdot$ SEM$^\text{L\&M}$')
	legend(*method_labels, location='outer north east')

	savefig(filepath)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
