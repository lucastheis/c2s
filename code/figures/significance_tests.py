import sys

sys.path.append('./code')

from numpy import sign, sum
from scipy.stats import wilcoxon
from comparison import *
from numpy.random import rand

def main(argv):
	print 'One-sided Wilcoxon signed rank test'
	print

	print 'Correlation'
	print

	for k, dataset in enumerate(datasets):
		print '   Dataset: {0}'.format(dataset_labels[k])

		for method1 in methods:
			for method2 in methods:
				if method1 != method2:
					corr1 = get_corr_all(eval('corr_{0}_{1}'.format(dataset, method1))).ravel()
					corr2 = get_corr_all(eval('corr_{0}_{1}'.format(dataset, method2))).ravel()

					# one-sided signed rank test
					W, p = wilcoxon(corr1, corr2)

					if sum(sign(corr1 - corr2)) < 0:
						p = 1. - p

					s = ''
					if p < 0.05:
						s = '(*)'
					if p < 0.01:
						s = '(**)'

					print '      {0} > {1}: {2:.4f} {3}'.format(method1, method2, p, s)

			print

	print
	print 'Information gain'
	print

	for k, dataset in enumerate(datasets):
		print '   Dataset: {0}'.format(dataset_labels[k])

		for method1 in methods:
			for method2 in methods:
				if method1 != method2:
					info1 = get_info_all(eval('lik_{0}_{1}'.format(dataset, method1))).ravel()
					info2 = get_info_all(eval('lik_{0}_{1}'.format(dataset, method2))).ravel()

					# one-sided signed rank test
					W, p = wilcoxon(info1, info2)

					if sum(sign(info1 - info2)) < 0:
						p = 1. - p

					s = ''
					if p < 0.05:
						s = '(*)'
					if p < 0.01:
						s = '(**)'

					print '      {0} > {1}: {2:.4f} {3}'.format(method1, method2, p, s)

			print

	print
	print 'Area under curve'
	print

	for k, dataset in enumerate(datasets):
		print '   Dataset: {0}'.format(dataset_labels[k])

		for method1 in methods:
			for method2 in methods:
				if method1 != method2:
					auc1 = get_auc_all(eval('auc_{0}_{1}'.format(dataset, method1))).ravel()
					auc2 = get_auc_all(eval('auc_{0}_{1}'.format(dataset, method2))).ravel()

					# one-sided signed rank test
					W, p = wilcoxon(auc1, auc2)

					if sum(sign(auc1 - auc2)) < 0:
						p = 1. - p

					s = ''
					if p < 0.05:
						s = '(*)'
					if p < 0.01:
						s = '(**)'

					print '      {0} > {1}: {2:.4f} {3}'.format(method1, method2, p, s)

			print

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
