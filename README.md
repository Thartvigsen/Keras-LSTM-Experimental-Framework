# An experimental framework for running LSTM experiments.

This Python code was used for the WPI Data Science REU Machine Learning for Healthcare Group. The framework is focused on LSTM's, particularly for training on Multivariate Clinical Time Series.

The *run.sh* file controls the hyperparameters for the LSTMs, and the *main.py* file controls the pipeline of data processing from a raw 3D tensor in the shape of (instances, timesteps, variables) to final predictions with multiple evaluation metrics.

Since this summer, I have moved on to work on a more general TensorFlow experimental framework.
