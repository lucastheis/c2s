# c2s

Tools for the prediction of spike trains from calcium traces.

## Example

Once installed, predicting spikes can be as easy as

	$ c2s predict data.mat predictions.mat

This package comes with a default model for predicting spikes from calcium traces, but also comes
with tools for training and evaluating your own model.

## Documentation

If you are a neuroscientist and want to reconstruct spikes from fluorescence/calcium traces or
similar signals, please see the main [__documentation__](http://c2s.readthedocs.org/en/latest/).
If you are a developer and want to use `c2s` in your own Python code, please see the [__API documentation__](http://lucastheis.github.io/c2s/).


## Requirements

* Python >= 2.7.0
* [cmt](https://github.com/lucastheis/cmt/) >= 0.5.0
* NumPy >= 1.6.1
* SciPy >= 0.13.0

## Installation

First make sure that you have the [Conditional Modeling Toolkit](https://github.com/lucastheis/cmt/) installed. Then run:

	pip install git+https://github.com/lucastheis/c2s.git

## References

If you use our code in your research, please cite the following paper:

L. Theis, P. Berens, E. Froudarakis, J. Reimer, M. Roman-Roson, T. Baden, T. Euler, A. S. Tolias, et al.  
[Supervised learning sets benchmark for robust spike detection from calcium imaging signals](http://bethgelab.org/publications/127/)  
bioRxiv, 2014
