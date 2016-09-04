import sys

def convert(input):
	"""
	Convert dictionary keys/values into something that can be saved with `scipy.io.savemat`.
	"""

	if isinstance(input, dict):
		return dict([(convert(key), convert(value)) for key, value in input.items()])
	elif isinstance(input, list):
		return [convert(element) for element in input]
	elif sys.version_info < (3,) and isinstance(input, unicode):
		return input.encode('utf-8')
	else:
		return input
