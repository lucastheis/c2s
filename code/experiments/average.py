"""
Averages responses over multiple experiments with different training/validation splits.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from glob import glob
from numpy import corrcoef, hstack
from scipy.io import loadmat, savemat
from tools import Experiment
from cmt.models import GLM, STM

cells = [5, 7, 48, 19, 50, 12, 17, 18, 28]

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--model', '-m', type=str, default='STM')

	args = parser.parse_args(argv[1:])

	files = glob('results/predictions.*.xpck')

	# load data
	data = loadmat('data/data.mat')['Data'].ravel()

	outputs = []
	outputs_test = []

	for i in range(data.shape[0]):
		outputs.append(data[i][1].reshape(1, -1))

		if i + 1 not in cells:
			outputs_test.append(outputs[-1])

	# average predictions
	counter = 0.

	predictions = 0
	predictions_test = 0

	correlations = []
	correlations_test = []

	model_type = GLM if args.model.upper() == 'GLM' else STM

	for filepath in files:
		results = Experiment(filepath)

		if isinstance(results['model'], model_type):
			predictions      = predictions      + hstack(results['predictions'])
			predictions_test = predictions_test + hstack(results['predictions_test'])
			counter += 1.

	predictions /= counter
	predictions_test /= counter

	corr      = corrcoef(hstack(outputs),      predictions)[0, 1]
	corr_test = corrcoef(hstack(outputs_test), predictions_test)[0, 1]

	print 'Average correlation:'
	print '\t{0:.5f} (test)'.format(corr_test)
	print '\t{0:.5f} (total)'.format(corr)

	print 'Correlation of average response:'
	print '\t{0:.5f} (test)'.format(corr_test)
	print '\t{0:.5f} (total)'.format(corr)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
