# Explainable Artificial Intelligence (XAI) for Green Technology Adoption

This repository provides the code and the data supporting the paper "Revealing drivers and barriers in the diffusion of rooftop photovoltaics using explainable AI" by Dorothea Kistinger, Maurizio Titz, Michael T. Schaub, Sandra Venghaus, and Dirk Witthaut.

We implemented the analysis in Python. All requirements are given in *requirements.txt*.

## Installation 

We implemented the analysis in Python. All requirements are given in *requirements.txt*.

## Data

### Raw Data

Three publicly available data sources where used in this project. 

1. Marktstammdatenregister: data on PV distribution provided by Germany's Bundesnetzagentur. It is available at this [link]()
2. INKAR: Socio-Economic data provided by the 
3. KBA: data on the distribution of different types of vehicles (BEV, combustible engines, ..) provided by the Kraftfahrtbundesamt (German Federal Motor Vehicles and Transport Authority) which is available at this [link]().
4. Voting data: results of local elections provided by which is available at ()

The data is supposed to be placed in raw_data ??

### Preprocessing Data
The scripts and jupyter notebooks to preprocess the raw data can be found in submodule [preprocessing](./xai_green_tech_adoption/preprocessing/). 
The preprocessed data will be placed in the folder ??, which is the input for the subsequent analysis using XAI (e.g., **gradient boosted tree models** and **Shapely Additive Values**).

## XAI

The pipelines

### Visualizing the results

The plots presented in the article that is currently under review, can be found in [plots_paper.py](./xai_green_tech_adoption/plots_paper.py). The last function is a meta function that produces all plots if required output data of the XAI pipeline was produced before.

