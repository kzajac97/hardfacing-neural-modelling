# hardfacing-neural-modlling

## Modelling of hardfacing layers deposition parameters using robust machine learning algorithms

Paper published in *Journal of Physics: Conference Series* and can be found under
[this link.](https://iopscience.iop.org/article/10.1088/1742-6596/2130/1/012016)

### Abstract

The study presents a data-driven framework for modelling parameters of hardfacing
deposits by GMAW using neural models to estimate the influence of process parameters without
the need of creating experimental samples of the material and detailed measurements. The
process of GAS Metal Arc Welding (GMAW) hardfacing does sometimes create nonhomogenous structures in the material
not only in deposited material, but also in the heat-affected
zone (HAZ) and base material. Those structures are not fully deterministic, so the modelling
method should account for this unpredictable component and only learn the generic structure of
the hardness of the resulting material. Artificial neural networks (ANN) were used to create a
model of the process using only measured samples without any knowledge of equations
governing the process. Robust learning was used to decrease the influence of outliers and noise
in the measured data on the neural model performance. The proposed method relies on
modification of the loss function and several of them are compared and evaluated as an attempt
to construct general framework for analysing the hardness as a function of electric current and
arc velocity. The proposed method can create robust models of the hardfacing layers deposition
or other welding processes and predict the properties of resulting materials even for unseen
parameters based on experimental data. This modelling framework is not typically used for
metallurgy, and it requires further case studies to verify its generalisability

### Structure

This repository contains notebooks and utility code used to run experiments required for the paper.
It is organized in the following way:
* data-exploration.ipynb - contains data exploration and preprocessing 
* baselines.ipynb - contains experiments related to baseline models for the problem
* experiments.ipynb - contains experiments using neural model

### Libraries

Libraries and tools used to produce the code required by the experiment:
* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [tensorflow](https://www.tensorflow.org/)
* [wandb](https://wandb.ai/site)
