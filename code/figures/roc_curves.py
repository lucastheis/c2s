import sys

sys.path.append('./code')

from comparison import *
from pgf import *
from numpy import hstack, any, linspace
from scipy.io import loadmat
from pickle import load
from calcium import downsample
from roc import roc
from tools import Experiment

pred_AOD_STM = 'results/predictions.3.stm_ga.xpck'
pred_AOD_FOO = 'results/predictions.3.fast_oopsi_cv.mat'
pred_AOD_YAK = 'results/predictions.3.yaksi.mat'

methods = ['STM', 'FOO', 'YAK', 'RAW']
 
def main(argv):
	figure(sans_serif=True)

	for method in methods:
		if method == 'RAW':
			predictions = [entry['calcium'] for entry in data]
		else:
			filepath = eval('pred_AOD_{0}'.format(method))

			if filepath.endswith('.mat'):
				predictions = loadmat(filepath)['predictions'].ravel()
			else:
				results = Experiment(filepath)
				predictions = results['predictions']

				with open(results['args'].dataset) as handle:
					data = load(handle)

		ds = 4
		predictions = hstack(downsample(pred, ds) for pred in predictions)
		spikes = hstack(downsample(entry['spikes'], ds) for entry in data)

		# marks bins containing spikes
		mask = spikes > .5

		# collect positive and negative examples
		neg = predictions[-mask]
		pos = []

		# this is necessary because any bin can contain more than one spike
		while any(mask):
			pos.append(predictions[mask])
			spikes -= 1
			mask = spikes > .5
		pos = hstack(pos)

		# compute area under curve
		roc_curve = roc(pos, neg)[1]
		roc_curve = roc_curve[::roc_curve.size // 1000]

		plot(linspace(0, 1, roc_curve.size), roc_curve,
			line_width=2,
			color=eval('color_{0}'.format(method)))
		xlim(0, 1)
		ylim(0, 1)
		xlabel('False positive rate')
		ylabel('True positive rate')
		axis(width=6, height=6)
		box('off')

	savefig('figures/roc_curve.pdf')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
