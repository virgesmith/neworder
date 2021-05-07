---
title: 'neworder: A dynamic microsimulation framework for Python'
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
 - name: School of Law, University of Leeds
   index: 1date: 7 May 2021
bibliography: paper.bib
---

# Summary

Traditional microsimulation frameworks typically use a proprietary modelling language, often place restrictions on data formats, and vary in terms of efficiency or scalability. *neworder* provides an efficient, flexible, and scalable framework for implementing microsimulation models using standard python code. Being a framework, it has been designed with reusability and extensibility as primary motivations.

It is implemented in C++ for maximal performance and supports both serial and parallel execution. Particular attention has been paid to provision of powerful and flexible random number and timestepping functionality.

It is extensively documented, including numerous detailed examples. The package is available in the standard python repositories (PyPI, conda-forge) and also as a docker image.

# Statement of need

The *neworder* framework is designed to be as unrestrictive and flexible as possible, whilst still providing a solid foundation on which to implement a model, and a suite of useful tools. Being agnostic to data formats means that models can be easily integrated with other models and/or into workflows with rigid input and output data requirements.

It supports both serial and parallel execution modes, with the latter being used to distribute computations for large populations or to perform sensitivity or convergence analyses. *neworder* runs as happily on a desktop PC as it does on a HPC cluster.

*neworder* is inspired by MODGEN [1] and, to a lesser extent, the python-based LIAM2 [2] tool, and can be thought of as a powerful best-of-both-worlds hybrid of MODGEN and LIAM2.

Both MODGEN and LIAM2 require their models to be specified in proprietary languages (based on C++ and yaml, respectively), whereas our framework eliminates the extra learning curve as users simply define their models in standard python code.

Whilst MODGEN supports parallel execution, LIAM2 does not. MODGEN is very restrictive with input data (which must be defined within the model code) and output data (which is a SQL database). *neworder* supports parallel execution, thus having the scalability of MODGEN, but without any restrictions on data sources or formats.

Both MODGEN and LIAM2 require manual installation and configuration of an environment in order to develop models; *neworder* can simply be installed with one command.

The framework is comprehensively documented [3] and specifically provides detailed examples that are translations of MODGEN models from Belanger \& Sabourin [4] and Statistics Canada [5,6], demonstrating how *neworder* implementations can be both simpler and more performant (see the Mortality example [3]).

Part of the design ethos is not to reinvent the wheel and leverage the huge range of statistical functions in packages like \emph{numpy} and \emph{scipy}. However, functions are provided where there is a useful niche function or a major efficiency gain to be had. An example of the former are methods provided to sample extremely efficiently from non-homogeneous Poisson processes using the Lewis-Schedler algorithm [7], and the ability to perform Markov transitions \emph{in situ} in a \emph{pandas} dataframe, both of which result in at least a factor-of-ten performance gain.

\begin{figure}[h]
	\centering
	\includegraphics[width=0.9\textwidth]{mortality-100k}
	\caption{Sampling mortality: "Discrete" samples repeatedly at 1 year intervals, "Continuous" uses the Lewis-Shedler algorithm to sample the entire curve, with a tenfold performance improvement}
\end{figure}

Another important consideration in *neworder*'s design is reproducibility, especially with regard to random number generators. Inbuilt seeding strategies allow for fully deterministic execution and control over whether parallel processes should be correlated or uncorrelated. User-defined seeding strategies are also supported.


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
