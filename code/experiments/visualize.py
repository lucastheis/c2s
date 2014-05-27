import sys

sys.path.append('./code')

from tools import Experiment
from matplotlib.pyplot import *
from numpy import *
from numpy import round, max
from argparse import ArgumentParser

def main(argv):
	parser = ArgumentParser(argv[0])
	parser.add_argument('--results', type=str, default='results/predictions.01082013.160221.xpck')

	args = parser.parse_args(argv[1:])

	results = Experiment(args.results)

	ion()

	P = results['pre'].pre_in.T
	P = P / sqrt(sum(square(P), 0))
	
	nrm = sqrt(sum(square(results['model'].features), 0))
	features = results['pre'].inverse(results['model'].features / nrm)
#	features = dot(results['pre'].pre_in.T, results['model'].features / nrm)
#	features = dot(P, results['model'].features / nrm)
	weights = results['model'].weights * square(nrm)
	predictors = results['pre'].inverse(results['model'].predictors.T)
#	predictors = dot(results['pre'].pre_in.T, results['model'].predictors.T)
#	predictors = dot(P, results['model'].predictors.T)

	print results['model'].predictors.T.shape
	print results['pre'].pre_in.shape

	sampling_rate = 11.

	# visualize linear features
	figure(figsize=(6, 6))
	for k in range(results['model'].num_components):
		plot(predictors[:, k], linewidth=2.)
	legend(['Component {0}'.format(k + 1) for k in range(results['model'].num_components)])
	xlabel('{0:.1f} s'.format(predictors.shape[0] / float(sampling_rate)))
	xticks([])
	yticks([])

	savefig('linear.pdf', transparent=True)

	# visualize quadratic features
	for k in range(results['model'].num_components):
		figure(figsize=(6, 6))

		alpha = abs(weights[k]) / max(abs(weights[k]))

		for i in range(results['model'].num_features):
			if weights[k, i] < 0:
				plot(features[:, i], 'r', alpha=alpha[i], linewidth=2.)
			else:
				plot(features[:, i], 'b', alpha=alpha[i], linewidth=2.)

		title('Component {0}'.format(k + 1))
		xticks([])
		yticks([])
		xlabel('{0:.1f} s'.format(features.shape[0] / sampling_rate))

		savefig('quadratic.{0}.pdf'.format(k + 1), transparent=True)

	# weights
	print round(weights, 2)

	raw_input()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
