"""
Plots linear and quadratic filters of an STM.
"""

import sys

sys.path.append('./code')

from numpy import dot, sqrt, max, min
from pgf import *
from tools import Experiment

# regularized
#filepath_AOD = 'results/visualization.3/train.08042014.171520.xpck'
#filepath_EUL = 'results/visualization.4/train.08042014.172252.xpck'

# no regularization
filepath_AOD = 'results/visualization.3/train.24042014.154729.xpck'
filepath_EUL = 'results/visualization.4/train.24042014.154153.xpck'

datasets = ['AOD', 'EUL']
dataset_labels = ['V1/OGB1', 'Retina/OGB1']

fps = 100.

def main(argv):
	figure(sans_serif=True)

	for i, dataset in enumerate(datasets):
		results = Experiment(eval('filepath_{0}'.format(dataset)))

		models = results['models']

		stm = models['models'][0]
		pca = models['pca']

		# LINEAR FEATURES
		subplot(len(datasets) - 1 - i, 0, spacing=3)

		fmax = 0.
		fmin = 0.

		for k in range(stm.num_components):
			gray = sqrt((stm.num_components - k - 1.) / (stm.num_components - 1.) * 0.64)
			filtr = dot(stm.predictors[[k]], pca.pre_in)

			plot(filtr.ravel(),
				color=RGB(gray, gray, gray),
				line_width=2)

			if max(filtr) > fmax:
				fmax = max(filtr)
			if min(filtr) < fmin:
				fmin = min(filtr)

		m = fmin + .90 * (fmax - fmin)
		n = fmin + .95 * (fmax - fmin)
		plot([fps / 5. * 0.4, fps / 5. * 1.4], [m, m], 'k-', line_width=2)
		text(fps / 5. * 0.4 + fps / 10., n, '\\small {0} ms'.format(int(results['args'].window_length) / 5))
		xlim(0, fps)
		ylim(fmin * 1.1, fmax * 1.1)
		t = int(results['args'].window_length / 2)
		xtick([0, fps / 2, fps], ['{0}'.format(-t), '0', '{0}'.format(t)])
		ytick([])
		box('off')
		ylabel('Linear features')
		xlabel('Time [ms]')
		title(dataset_labels[i])
		axis(width=5, height=5)

		# QUADRATIC FEATURES
		subplot(len(datasets) - 1 - i, 1)

		fmax = 0.
		fmin = 0.

		for k in range(stm.num_features):
			gray = sqrt((stm.num_features - k - 1.) / (stm.num_features - 1.) * 0.64)
			filtr = dot(stm.features[:, [k]].T, pca.pre_in)

			plot(filtr.ravel(),
				color=RGB(gray, gray, gray),
				line_width=2)

			if max(filtr) > fmax:
				fmax = max(filtr)
			if min(filtr) < fmin:
				fmin = min(filtr)

		m = fmin + .90 * (fmax - fmin)
		n = fmin + .95 * (fmax - fmin)
		plot([fps / 5. * 0.4, fps / 5. * 1.4], [m, m], 'k-', line_width=2)
		text(fps / 5. * 0.4 + fps / 10., n, '\\small {0} ms'.format(int(results['args'].window_length) / 5))
		xlim(0, fps)
		ylim(fmin * 1.1, fmax * 1.1)
		t = int(results['args'].window_length / 2)
		xtick([0, fps / 2, fps], ['{0}'.format(-t), '0', '{0}'.format(t)])
		ytick([])
		box('off')
		xlabel('Time [ms]')
		ylabel('Quadratic features')
		title(dataset_labels[i])
		axis(width=5, height=5)

		subplot(len(datasets) - 1 - i, 1)

	savefig('figures/filters.pdf')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
