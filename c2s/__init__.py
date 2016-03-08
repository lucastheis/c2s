__all__ = ['load_data', 'preprocess', 'train', 'evaluate', 'predict']
from .c2s import __version__
from .c2s import load_data, preprocess, train, predict, evaluate
from .c2s import percentile_filter, downsample, robust_linear_regression
