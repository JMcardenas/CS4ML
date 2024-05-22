# CS4ML: A general framework for active learning with arbitrary data based on Christoffel functions 

Christoffel Sampling for Machine Learning is a general framework for active learning in regression problems. It approximates a target function arising from general types of data, rather than pointwise samples. 

This generalization covers many cases of practical interest, such as data acquired in transform domains (e.g., Fourier data), vector-valued data (e.g., gradient-augmented data), data acquired along continuous curves, and, multimodal data, i.e., combinations of different types of measurements. 

This library shows the efficacy of our framework for gradient-augmented learning with polynomials, Magnetic Resonance Imaging (MRI) using generative models and adaptive sampling for solving PDEs using Physics-Informed Neural Networks (PINNs), see spotlight presentation at [spotlight](https://neurips.cc/virtual/2023/poster/71203). You can use the tools provided in this repository for your own data. 
 
This is a repository associated with the paper:

_CS4ML: A general framework for active learning with arbitrary data based on Christoffel functions_ by Juan M. Cardenas, Ben Adcock and Nick Dexter  

published by NeurIPS 2023, available at https://openreview.net/pdf?id=aINqoP32cb .

If you have questions or comments about the code, please contact [juan.cardenascardenas@colorado.edu](mailto:juan.cardenascardenas@colorado.edu?subject=[GitHub]%20Source%20Han%20Sans), [ben_adcock@sfu.ca](mailto:ben_adcock@sfu.ca?subject=[GitHub]%20Source%20Han%20Sans), or [nick.dexter@fsu.edu](mailto:nick.dexter@fsu.edu?subject=[GitHub]%20Source%20Han%20Sans).

Parts of this repository are based on the code for the paper "CAS4DL: Christoffel Adaptive Sampling for function approximation via Deep Learning" by Ben Adcock, Juan M. Cardenas and Nick Dexter,  which is available in the repository [https://github.com/JMcardenas/CAS4DL](https://github.com/JMcardenas/CAS4DL).

# Code organization 
Files are organized into three main directories:

## Gradient-Augmented-with-polynomials 
Contains the main Matlab files used to create figures in each section

Organized in sections

## Generative-models  
Contains various python functions needed across main scripts

## Physic-Informed-Neural-Networks with CAS 
Contains .mat files generated by scripts in src

Organized in sections

## Filenames

These generaically take the form 

fig_[sec][number]\_[row]_[col]

where [sec] is the section number, [number], [number1] and [number2] are the figure numbers and [row] and [col] are the row number and column number is multi-panel figures.

# Installation and Quick Start 

You can clone the repository and install locally. It is highly recommended to clone in editable mode so the changes are applied without having to reinstall:
 
```
git clone https://github.com/JMcardenas/CS4ML.git
```
Next, run mainiter.py or modify config0.py to your own purposes: 

```
run main_iter.py
```

# Citing 
In case you use CS4ML in academic papers or scientific reports, please go ahead and cite:

```
@inproceedings{
cardenas2023csml,
title={{CS}4{ML}: A general framework for active learning with arbitrary data based on Christoffel functions},
author={Juan M Cardenas and Ben Adcock and Nick Dexter},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=aINqoP32cb}
}
```
