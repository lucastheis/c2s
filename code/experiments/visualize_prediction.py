import sys

sys.path.append('./code')

from numpy import *
from pgf import *
from scipy.io import loadmat

def main(argv):
	data = loadmat('data/data.mat')['Data'].ravel()

	results = loadmat('results/predictions.STM.mat')

	# find best cell
	i = argmax(results['corr'])

	galvo_trace = data[i][0].ravel()
	spike_trace = data[i][1].ravel()
	sampling_rate = float(data[i][2])

	start = 1000
	stop = 3000

	# spikes
	subplot(1, 0)

	spike_times = where(spike_trace[start:stop])[0]
	stem(spike_times, spike_trace[start:stop][spike_times], 'k-', line_width=1.2, marker=None)

	predictions = results['predictions'][i, 0].ravel()
	plot(predictions[start:stop])

	legend('Observed', 'Predicted')

	ticks = linspace(0, stop - start, 11)
	ticklabels = ['{0:.1f}'.format(float(t) / sampling_rate) for t in ticks]

	xtick(ticks, ticklabels)
	xlabel('Time [s]')

	ylabel('Number of spikes')
	xlim(0, stop - start)
	gca().width = 20
	gca().height = 5

	# calcium
	subplot(0, 0)

	plot(galvo_trace[start:stop])

	xlim(0, stop - start)
	xtick(ticks, ticklabels)
	ytick([])
	title('Calcium')
	xlabel('Time [s]')
	gca().width = 20
	gca().height = 5

	gcf().sans_serif = True

	draw()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
