#!/usr/bin/env python

from setuptools import setup

setup(
	name='calcium',
	version=__version__,
	author='Lucas Theis',
	author_email='lucas@theis.io',
	description='A toolbox for inferring spikes from two-photon imaging calcium traces.',
	url='https://github.com/lucastheis/calcium/',
	packages=find_packages(),
	include_package_data=True,
	install_requires=('cmt>=1.5.0'),
	license='MIT',
	classifiers=(
		'Development Status :: 2 - Pre-Alpha',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
		'Programming Language :: Python'),
)
