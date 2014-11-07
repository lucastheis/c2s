#!/usr/bin/env python

from setuptools import setup
from c2s import __version__
try:
	from Cython.Build import cythonize
except:
	cythonize = lambda _: []

setup(
	name='c2s',
	version=__version__,
	author='Lucas Theis',
	author_email='lucas@theis.io',
	description='A toolbox for inferring spikes from two-photon imaging calcium traces.',
	url='https://github.com/lucastheis/c2s/',
	packages=['c2s'],
	scripts=[
		'scripts/c2s',
		'scripts/c2s-preprocess.py',
		'scripts/c2s-predict.py',
		'scripts/c2s-train.py',
		'scripts/c2s-evaluate.py',
		'scripts/c2s-leave-one-out.py'],
	install_requires=(),
	license='MIT',
	zip_safe=False,
	classifiers=(
		'Development Status :: 2 - Pre-Alpha',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
		'Programming Language :: Python'),
	ext_modules=cythonize("c2s/roc.pyx"),
)
