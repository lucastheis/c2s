# c2s

![predictions](https://raw.githubusercontent.com/lucastheis/c2s/media/predictions.png)

Tools for the prediction of spike trains from calcium traces.

## Documentation

If you are a neuroscientist and want to reconstruct spikes from fluorescence/calcium traces or
similar signals, please see the main [__documentation__](http://c2s.readthedocs.org/en/latest/).
If you are a developer and want to use `c2s` in your own Python code, please see the [__API documentation__](http://lucastheis.github.io/c2s/).

## Example

Once installed, predicting spikes can be as easy as

	$ c2s predict data.mat predictions.mat

This package comes with a default model for predicting spikes from calcium traces, but also comes
with tools for training and evaluating your own model.

## Requirements

* Python >= 2.7.0
* [cmt](https://github.com/lucastheis/cmt/) >= 0.5.0
* NumPy >= 1.6.1
* SciPy >= 0.13.0
* Cython >= 0.20.0 (optional)
* Matplotlib >= 1.4.2 (optional)

## Installation

First install the [Conditional Modeling Toolkit](https://github.com/lucastheis/cmt/). Then run:

	$ pip install git+https://github.com/lucastheis/c2s.git

You can avoid manually installing c2s and its requirements by using
[Docker](https://www.docker.com). A Dockerfile for c2s is provided
by [Jonas Rauber](https://github.com/jonasrauber/c2s-docker). This might make your life
easier especially if you are planning to use Windows or Mac OS.

## References

If you use our code in your research, please cite the following paper:

L. Theis, P. Berens, E. Froudarakis, J. Reimer, M. Roman-Roson, T. Baden, T. Euler, A. S. Tolias, et al.  
[Benchmarking spike rate inference in population calcium imaging](https://bethgelab.org/publication/2016_05_theis/)  
Neuron, 90(3), 471-482, 2016

The default model was trained on many datasets (together containing roughly 110,000 spikes) from
different labs. Therefore, if you use the default model for prediction, please also cite:

J. R. Cotton, E. Froudarakis, P. Storer, P. Saggau, and A. S. Tolias  
Three-dimensional mapping of microcircuit correlation structure  
Frontiers in Neural Circuits, 2013

J. Akerboom et al.  
Optimization of a GCaMP calcium indicator for neural activity imaging  
Journal of Neuroscience, 2012

T. W. Chen et al.  
Ultrasensitive fluorescent proteins for imaging neuronal activity  
Nature, 2013
