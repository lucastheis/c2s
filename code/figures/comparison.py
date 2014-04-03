"""
Generates bar plots of correlations and information rates.
"""

import os
import sys

sys.path.append('./code')

from numpy import argmin, abs, mean, std, sqrt, square, asarray, zeros, ones, triu
from pgf import *
from pgf.colormap import colormaps
from tools import Experiment

filepath = 'figures/comparison.pdf'

# colors corresponding to the different methods
cmap = 'jet'
color_STM = RGB(0.0, 0.0, 0.0)
color_STX = RGB(0.0, 0.0, 0.0)
color_NNP = RGB(0.5, 0.5, 0.5)
color_LNP = RGB(0.8, 0.8, 0.8)
color_FOO = RGB(*colormaps[cmap].colors[80])
color_FOX = RGB(*colormaps[cmap].colors[80])
color_FO2 = RGB(*colormaps[cmap].colors[90])
color_SOO = RGB(*colormaps[cmap].colors[120])
color_RAW = RGB(*colormaps[cmap].colors[180])
color_YAK = RGB(*colormaps[cmap].colors[220])

# locations of precomputed correlations
corr_AOD_STM = 'results/correlations.3.stm_ga.xpck'
corr_AOD_STX = 'results/correlations.3.stmx_1.xpck'
corr_AOD_NNP = 'results/correlations.3.nnp_10_5.xpck'
corr_AOD_LNP = 'results/correlations.3.lnp.xpck'
corr_AOD_FOO = 'results/correlations.3.fast_oopsi_cv.xpck'
corr_AOD_FOX = 'results/correlations.3.fast_oopsi.xpck'
corr_AOD_SOO = 'results/correlations.3.smc_oopsi.xpck'
corr_AOD_YAK = 'results/correlations.3.yaksi.xpck'
corr_AOD_RAW = 'results/correlations.3.raw.xpck'

corr_EUL_STM = 'results/correlations.4.stm.xpck'
corr_EUL_STX = 'results/correlations.4.stmx_1.xpck'
corr_EUL_NNP = 'results/correlations.4.nnp_5_3.xpck'
corr_EUL_LNP = 'results/correlations.4.lnp.xpck'
corr_EUL_FOO = 'results/correlations.4.fast_oopsi_cv.xpck'
corr_EUL_FOX = 'results/correlations.4.fast_oopsi.xpck'
corr_EUL_SOO = 'results/correlations.4.smc_oopsi.xpck'
corr_EUL_YAK = 'results/correlations.4.yaksi.xpck'
corr_EUL_RAW = 'results/correlations.4.raw.xpck'

# locations of precomputed likelihoods
lik_AOD_STM = 'results/likelihoods.3.stm_ga.xpck'
lik_AOD_STX = 'results/likelihoods.3.stmx_1.xpck'
lik_AOD_NNP = 'results/likelihoods.3.nnp_10_5.xpck'
lik_AOD_LNP = 'results/likelihoods.3.lnp.xpck'
lik_AOD_FOO = 'results/likelihoods.3.fast_oopsi_cv.xpck'
lik_AOD_FOX = 'results/likelihoods.3.fast_oopsi.xpck'
lik_AOD_SOO = 'results/likelihoods.3.smc_oopsi.xpck'
lik_AOD_YAK = 'results/likelihoods.3.yaksi.xpck'
lik_AOD_RAW = 'results/likelihoods.3.raw.xpck'

lik_EUL_STM = 'results/likelihoods.4.stm.xpck'
lik_EUL_STX = 'results/likelihoods.4.stmx_1.xpck'
lik_EUL_NNP = 'results/likelihoods.4.nnp_5_3.xpck'
lik_EUL_LNP = 'results/likelihoods.4.lnp.xpck'
lik_EUL_FOO = 'results/likelihoods.4.fast_oopsi_cv.xpck'
lik_EUL_FOX = 'results/likelihoods.4.fast_oopsi.xpck'
lik_EUL_SOO = 'results/likelihoods.4.smc_oopsi.xpck'
lik_EUL_YAK = 'results/likelihoods.4.yaksi.xpck'
lik_EUL_RAW = 'results/likelihoods.4.raw.xpck'

datasets = ['AOD', 'EUL']
dataset_labels = ['V1/OGB1', 'Retina/OGB1']
#methods = ['STM', 'NNP', 'LNP']
#method_labels = ['STM', 'MLP', 'LNP']
methods = ['STM', 'FOO', 'YAK', 'RAW']
method_labels = ['STM', 'Vogelstein et al. (2010)', 'Yaksi \& Friedrich (2006)', 'Raw']
#methods = ['STX', 'FOX', 'RAW']
#method_labels = ['STM$^*$', 'Vogelstein et al. (2010)', 'Raw']

def get_corr(filepath, fps=25.):
	"""
	Extracts average correlation at given sampling rate from experiment.
	"""

	if not os.path.exists(filepath):
		print filepath, 'does not exist.'
		return 0., 0.

	results = Experiment(filepath)

	idx = argmin(abs(mean(results['fps'], 1) - fps))

	N = asarray(results['correlations']).shape[1]

	corr = mean(results['correlations'], 1)[idx]
	sem = std(results['correlations'], 1, ddof=1)[idx] / sqrt(N)

	return corr, sem



def get_corr_all(filepath, fps=25.):
	"""
	Extracts all correlation at given sampling rate from experiment.
	"""

	results = Experiment(filepath)
	idx = argmin(abs(mean(results['fps'], 1) - fps))
	return results['correlations'][idx]



def get_info(filepath, fps=25.):
	"""
	Extracts average information rate at given sampling rate from experiment.
	"""

	if not os.path.exists(filepath):
		print filepath, 'does not exist.'
		return 0., 0.

	results = Experiment(filepath)

	idx = argmin(abs(mean(results['fps'], 1) - fps))

	info = asarray(results['entropy']) + asarray(results['loglik'])
	sem = std(info, 1, ddof=1)[idx] / sqrt(info.shape[1])
	info = mean(info, 1)[idx]

	return info, sem



def get_info_all(filepath, fps=25.):
	"""
	Extracts all information rates at given sampling rate from experiment.
	"""

	results = Experiment(filepath)
	idx = argmin(abs(mean(results['fps'], 1) - fps))
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

	xtick(xval, dataset_labels)
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

	xtick(xval, dataset_labels)
	axis(
		xmin=0.5,
		xmax=len(datasets) + .5,
		ymin=0.,
		width=2.5 * len(datasets),
		height=5)
	box('off')
	ylabel(r'Information gain $\pm$ 2 $\cdot$ SEM$^\text{L\&M}$ [bit/s]')
	legend(*method_labels, location='outer north east')

	savefig(filepath)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
