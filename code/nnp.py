import theano as th
import theano.tensor as tt

from numpy import hstack, sqrt
from numpy.random import rand, randn
from scipy.optimize import minimize
from cmt.models import ConditionalDistribution

def distribute(params, values):
	"""
	Assigns parameter values stored in a single array to the given Theano variables.
	"""

	values = values.ravel()

	for param in params:
		param.set_value(values[:param.size.eval()].reshape(param.shape.eval()))
		values = values[param.size.eval():]



def collect(params):
	"""
	Turns many shared Theano variables into one long array.
	"""

	return hstack(param.get_value().ravel() for param in params)



def split(lst, sizes):
	"""
	Splits list into chunks of specifized sizes.
	"""

	chunks = []
	for size in sizes:
		chunks.append(lst[:size])
		lst = lst[size:]
	return chunks



class NNP(object):
	"""
	Neural network with Poisson outputs.
	"""

	def __init__(self, num_inputs, num_hiddens=[]):
		self.num_inputs = num_inputs
		self.num_hiddens = num_hiddens

		# container for parameters
		self.A = []
		self.b = []

		# nodes representing inputs and outputs
		self.x = tt.dmatrix()
		self.y = self.x

		N = self.num_inputs

		for M in num_hiddens:
			# create parameters of intermediate layer
			self.A.append(th.shared(randn(M, N) / sqrt(N)))
			self.b.append(th.shared(rand(M, 1), broadcastable=(False, True)))

			# hidden layer activity
			self.y = tt.maximum(tt.dot(self.A[-1], self.y) + self.b[-1], 0)

			N = M

		# create parameters of output layer
		self.A.append(th.shared(randn(1, N) / sqrt(N)))
		self.b.append(th.shared(rand(1, 1) - 1., broadcastable=(False, True)))

		# output of the network
		self.y = tt.exp(tt.dot(self.A[-1], self.y) + self.b[-1])
		
		# objective function (negative Poisson log-likelihood)
		self.k = tt.dmatrix()
		self.E = tt.mean(self.y - self.k * tt.log(self.y))

		# gradients
		self.gA = tt.grad(self.E, self.A)
		self.gb = tt.grad(self.E, self.b)
		
		# computes objective function and gradients
		self.f = th.function([self.x, self.k], [self.E] + self.gA + self.gb)

		# negative Poisson log-likelihood
		self.loglikelihood = th.function([self.x, self.k], self.k * tt.log(self.y) - tt.gammaln(self.k + 1) - self.y)



	def train(self, inputs, outputs, parameters={}):
		parameters.setdefault('verbosity', 1)
		parameters.setdefault('max_iter', 100)

		def f(x):
			# set parameter values
			distribute(self.A + self.b, x)
			
			# compute objective function value and gradients
			E, gA, gb = split(
				self.f(inputs, outputs), [1, len(self.A), len(self.b)])

			# return value and packed gradients
			return E[0], hstack(param.ravel() for param in gA + gb)

		# optimize neural network
		res = minimize(f, collect(self.A + self.b),
			jac=True,
			method='L-BFGS-B',
			options={
				'disp': 1,
				'maxiter': parameters['max_iter'],
				'iprint': parameters['verbosity']})

		# set parameters
		distribute(self.A + self.b, res.x)



	def predict(self, inputs):
		"""
		Compute the average output for the given inputs.
		"""

		return self.y.eval({self.x: inputs})


	def evaluate(self, inputs, outputs):
		return -mean(self.loglikelihoods(inputs, outputs)) / log(2.)
