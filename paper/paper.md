---
title: 'neworder: a dynamic microsimulation framework for Python'
tags:
  - Python
  - Pybind11
  - C++
  - distributed computing
  - microsimulation
  - Monte-Carlo simulation
authors:
  - name: Andrew P Smith
    orcid: 0000-0002-9951-6642
    affiliation: 1
affiliations:
 - name: School of Law, University of Leeds, UK
   index: 1
date: 8 May 2021
bibliography: paper.bib
---

## Summary

Traditional microsimulation frameworks typically use a proprietary modelling language, often place restrictions on data formats, and vary in terms of efficiency or scalability. *neworder* provides an efficient, flexible, and scalable framework for implementing microsimulation models using standard Python code. Being a framework, it has been designed with reusability and extensibility as primary motivations.

It is predominantly implemented in C++ for maximal performance and supports both serial and parallel execution. Particular attention has been paid to the provision of powerful and flexible random number generation and timestepping functionality.

The package is extensively documented, including numerous detailed examples that showcase the functionality across a diverse range of applications including demography, finance, physics, and ecology.

It is available through the standard Python repositories (PyPI, conda-forge) and also as a Docker image.

## Statement of Need

The *neworder* framework is designed to be as unrestrictive and flexible as possible, whilst still providing a solid foundation on which to implement a microsimulation or individual-based model. Being agnostic to data formats means that models can be easily integrated with other models and/or into workflows with rigid input and output data requirements.

It supports both serial and parallel execution modes, with the latter using MPI to distribute computations for large populations or to perform sensitivity or convergence analyses. *neworder* runs as happily on a desktop PC as it does on a HPC cluster.

*neworder* was inspired by MODGEN [@government_of_canada_modgen_2009-1] and, to a lesser extent, the Python-based LIAM2 [@noauthor_liam2_nodate] tool, and can be thought of as a powerful best-of-both-worlds hybrid of MODGEN and LIAM2.

Both MODGEN and LIAM2 require their models to be specified in proprietary languages (based on C++ and YAML, respectively), whereas our framework eliminates the extra learning curve as users simply define their models in standard Python code.

Whilst MODGEN supports parallel execution, LIAM2 does not. MODGEN is very restrictive with input data (which must be defined within the model code) and output data (which is a SQL database). *neworder* supports parallel execution, thus having the scalability of MODGEN, but without any restrictions on data sources or formats.

Both MODGEN and LIAM2 require manual installation and configuration of an environment in order to develop models; *neworder* and its dependencies can simply be installed with a single command.

The framework is comprehensively documented [@smith_neworder_2021] and specifically provides detailed examples that are translations of MODGEN models from @belanger_microsimulation_2017 and Statistics Canada [@government_of_canada_general_2009, @government_of_canada_modgen_2009], demonstrating how *neworder* implementations can be both simpler and more performant (see the Mortality example in the documentation).

Part of the design ethos is not to reinvent the wheel and to leverage the huge range of statistical functions in packages like *numpy* and *scipy*. However, functions are provided where there is a useful niche function or a major efficiency gain to be had. An example of the former are methods provided to sample extremely efficiently from non-homogeneous Poisson processes using the Lewis-Shedler algorithm [@lewis_simulation_1979], and the ability to perform Markov transitions *in situ* in a pandas dataframe, both of which result in at least a factor-of-ten performance gain.

![Sampling mortality: "Discrete" samples repeatedly at 1 year intervals, "Continuous" uses the Lewis-Shedler algorithm to sample the entire curve, with a tenfold performance improvement.\label{fig:mortality-example}](mortality-100k.png)

Another important consideration in *neworder*'s design is reproducibility, especially with regard to random number generators. Inbuilt extensible seeding strategies allow for fully deterministic execution and control over whether parallel processes should be correlated or uncorrelated, and users can implement their own custom strategies as necessary.

*neworder* is currently being used for a project developing an integrated supply-demand model of police and crime [@noauthor_m-o-p-dpolice-supply-demand_2021]: a microsimulation of crime at high spatial, temporal and categorical resolution drives an agent-based model of police resourcing (implemented in netlogo), which in turn can dynamically alter the microsimulation parameters according to how well it responds to the events generated.

## Acknowledgements

This project is currently supported by Wave 1 of The UKRI Strategic Priorities Fund under the EPSRC Grant EP/T001569/1 and administered through the Alan Turing Institute.

## References
