import os
import sys

sys.path.append('./code')

from collections import defaultdict
from argparse import ArgumentParser
from numpy import asarray, ones, mean, argmin, abs, sort, std, percentile, any, median
from glob import glob
from pgf import *
from tools import Experiment

colors = [RGB(0, 0, 0), RGB(0.4, 0.4, 0.4), RGB(0.6, 0.6, 0.6)]
line_styles = ['-', '---', '--']
results = ['results/training_set_size.3', 'results/training_set_size.6', 'results/training_set_size.4']
labels = ['V1/OGB1/AOD', 'V1/OGB1/Glv', 'Retina/OGB1/Glv']

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--fps',     '-f', type=float, default=25.)

	args = parser.parse_args(argv[1:])

	figure(sans_serif=True, margin=3)

	for k, resultspath in enumerate(results):
		corr = defaultdict(lambda: [])
		auc = defaultdict(lambda: [])
		info = defaultdict(lambda: [])
		info_relative = defaultdict(lambda: [])

		print 'Collecting correlation results...'

		for filepath in glob(os.path.join(resultspath, '*correlations.xpck')):
			res_corr = Experiment(filepath)
			res_pred = Experiment(res_corr['args'].results)

			correlations = asarray(res_corr['correlations'])

			# index of correlations at given sampling rate
			idx = argmin(abs(mean(res_corr['fps'], 1) - args.fps))

			mask = ones(correlations.shape[1], dtype=bool)
			mask[res_pred['training_cells']] = False

			corr[res_pred['args'].num_train].extend(correlations[idx][mask])

		print 'Collecting information gain results...'

		for filepath in glob(os.path.join(resultspath, '*likelihoods.xpck')):
			res_info = Experiment(filepath)
			res_pred = Experiment(res_info['args'].results)

			# index of information gains at given sampling rate
			idx = argmin(abs(mean(res_info['fps'], 1) - args.fps))

			information = asarray(res_info['entropy'][idx]) + asarray(res_info['loglik'][idx])
			information_relative = information / asarray(res_info['entropy'][idx])

			if any(information < -100):
				print filepath

			mask = ones(information.size, dtype=bool)
			mask[res_pred['training_cells']] = False

			info[res_pred['args'].num_train].extend(information[mask])
			info_relative[res_pred['args'].num_train].extend(information_relative[mask])

		print 'Collecting area under curve results...'

		for filepath in glob(os.path.join(resultspath, '*auc.xpck')):
			res_auc = Experiment(filepath)
			res_pred = Experiment(res_auc['args'].results)

			aucs = asarray(res_auc['auc'])

			# index of ROC scores at given sampling rate
			idx = argmin(abs(mean(res_auc['fps'], 1) - args.fps))

			mask = ones(aucs.shape[1], dtype=bool)
			mask[res_pred['training_cells']] = False

			auc[res_pred['args'].num_train].extend(aucs[idx][mask])

		num_cells = sort(corr.keys())

		corr_avg = []
		corr_std = []
		corr_05p = []
		corr_95p = []

		info_avg = []
		info_std = []
		info_05p = []
		info_95p = []

		info_relative_avg = []

		auc_avg = []
		auc_std = []
		auc_05p = []
		auc_95p = []

		for n in num_cells:
			corr_avg.append(mean(corr[n]))
			corr_std.append(std(corr[n], ddof=1))
			corr_05p.append(percentile(corr[n], 5))
			corr_95p.append(percentile(corr[n], 95))

			info_avg.append(mean(info[n]))
			info_std.append(std(info[n], ddof=1))
			info_05p.append(percentile(info[n], 5))
			info_95p.append(percentile(info[n], 95))

			info_relative_avg.append(mean(info_relative[n]))

			auc_avg.append(mean(auc[n]))
			auc_std.append(std(auc[n], ddof=1))
			auc_05p.append(percentile(auc[n], 5))
			auc_95p.append(percentile(auc[n], 95))

		corr_avg = asarray(corr_avg)
		corr_std = asarray(corr_std)
		corr_05p = asarray(corr_05p)
		corr_95p = asarray(corr_95p)

		info_avg = asarray(info_avg)
		info_std = asarray(info_std)
		info_05p = asarray(info_05p)
		info_95p = asarray(info_95p)

		info_relative_avg = asarray(info_relative_avg)

		auc_avg = asarray(auc_avg)
		auc_std = asarray(auc_std)
		auc_05p = asarray(auc_05p)
		auc_95p = asarray(auc_95p)

		print 'Plotting...'

		subplot(0, 0, spacing=3)

		plot(num_cells, corr_avg, line_styles[k % len(line_styles)], color=colors[k % len(colors)], line_width=2)
#		plot(num_cells, corr_05p, line_styles[k % len(line_styles)], color=colors[k % len(colors)], line_width=1, pgf_options=['forget plot'])
#		plot(num_cells, corr_95p, line_styles[k % len(line_styles)], color=colors[k % len(colors)], line_width=1, pgf_options=['forget plot'])
		xlabel('Number of training cells')
		ylabel('Correlation')
		ylim(0, .6)
		xlim(0, 20)
		ytick([0., 0.1, 0.2, 0.3, 0.4, 0.5])
		xtick([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
		axis(width=5, height=5, axis_on_top=False)
		box('off')
		grid('on')

		subplot(0, 1)

		plot(num_cells, auc_avg, line_styles[k % len(line_styles)], color=colors[k % len(colors)], line_width=2)
#		plot(num_cells, auc_05p, line_styles[k % len(line_styles)], color=colors[k % len(colors)], line_width=1, pgf_options=['forget plot'])
#		plot(num_cells, auc_95p, line_styles[k % len(line_styles)], color=colors[k % len(colors)], line_width=1, pgf_options=['forget plot'])
		xlabel('Number of training cells')
		ylabel('Area under curve')
		ylim(.5, 1.0)
		xlim(0, 20)
		xtick([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
		axis(width=5, height=5, axis_on_top=False)
		box('off')
		grid('on')

		subplot(1, 0)

		info_05p[info_05p < 0.] = 0.
		info_95p[info_95p < 0.] = 0.
		info_avg[info_avg < 0.] = 0.

		plot(num_cells, info_avg, line_styles[k % len(line_styles)], color=colors[k % len(colors)], line_width=2)
#		plot(num_cells, info_05p, line_styles[k % len(line_styles)], color=colors[k % len(colors)], line_width=1, pgf_options=['forget plot'])
#		plot(num_cells, info_95p, line_styles[k % len(line_styles)], color=colors[k % len(colors)], line_width=1, pgf_options=['forget plot'])
		xlabel('Number of training cells')
		ylabel('Information gain [bit/s]')
		ylim(0, 4.5)
		xlim(0, 20)
		xtick([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
		axis(width=5, height=5, axis_on_top=False)
		box('off')
		grid('on')

		subplot(1, 1)

		info_05p[info_05p < 0.] = 0.
		info_95p[info_95p < 0.] = 0.
		info_avg[info_avg < 0.] = 0.

		plot(num_cells, info_relative_avg, line_styles[k % len(line_styles)], color=colors[k % len(colors)], line_width=2)
		xlabel('Number of training cells')
		ylabel('Relative Information gain')
		ylim(0, 0.4)
		xlim(0, 20)
		xtick([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
		axis(width=5, height=5, axis_on_top=False)
		box('off')
		grid('on')

	legend(*labels, location='outer north east')

	savefig('figures/training_set_size.tex')
	savefig('figures/training_set_size.pdf')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
