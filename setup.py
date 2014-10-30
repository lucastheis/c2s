#!/usr/bin/env python

import sys

sys.path.append('./code')

from setuptools import setup
from calcium import __version__

setup(
	name='calcium',
	version=__version__,
	author='Lucas Theis',
	author_email='lucas@theis.io',
	description='A toolbox for inferring spikes from two-photon imaging calcium traces.',
	url='https://github.com/lucastheis/calcium/',
	packages=['calcium'],
	scripts=[],
	install_requires=('cmt>=1.5.0'),
	zip_safe=False,
	license='MIT',
	classifiers=(
		'Development Status :: 2 - Pre-Alpha',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
		'Programming Language :: Python'),
)
