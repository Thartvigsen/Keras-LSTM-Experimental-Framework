# An experimental framework for running LSTM experiments.

*This repository is no longer in use, I have since learned a lot and moved to PyTorch - Tom, 11.09.2018*

This Python code was used for the 2017 WPI Data Science REU Machine Learning for Healthcare Group. The framework is focused on LSTM's, particularly for training on Multivariate Clinical Time Series.

The *run.sh* file controls the hyperparameters for the LSTMs, and the *main.py* file controls the pipeline of data processing from a raw 3D tensor in the shape of (instances, timesteps, variables) to final predictions with multiple evaluation metrics.

Since this summer, I have moved on to work on a more general TensorFlow experimental framework.

All data used this summer has been removed from this directory, as it is not freely available to the public. However, the MIMIC III database can be downloaded after requesting access for free which can be used to extract such data for this framework.
