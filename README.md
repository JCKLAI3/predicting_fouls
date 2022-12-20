# Predicting the number of fouls in football matches

This problem was what I had to try to answer for my master's project. The dataset in /data was provided by a company called Smartodds https://www.smartodds.co.uk/. The aim of this project was to explore the different machine learning techniques and find the best combination for predicting the number of fouls in a football match.

# Prerequiites

Before you start, ensure you meet the following requirements:
* Downloaded Python version 3.10.8
* Downloaded pyenv 
* Downloaded poetry to your local machine

# Project layout

This project has it's main code in the src/ directory which contains multiple directories for stages of a modelling project life cycle. 

The etl/ directory contains scripts for the different pre-preprocessing tasks:
* encoding.py - contained functions to encode categorical data to numeric
* feature_scaling.py - contains functions to scale data to desired output
* feature_selection.py - contain functions used to extract the features to be used for modelling

models/ directory 
* functions.py - contains models used for project
* NB: models used are exported from sci-kit learn but an extension will include squeaking hyper-parameters and as well as writing models from scratch 

evaluation/ directory
* evaluation_metrics.py -  contains functions for evaluating performance of models
* NB: Metrics used from sci-kit learn.

### Notebooks

In the notebooks section, we have an example notebook ```notebooks/Comparing models.ipynb``` to demo the process of running multiple different models to evaluate the best combination of pre-processing techniques and models to achieve the best performing result. Factors such as number of features used are also included. 