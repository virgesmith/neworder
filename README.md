# neworder

[![Build Status](https://travis-ci.org/virgesmith/neworder.png?branch=master)](https://travis-ci.org/virgesmith/neworder) 
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)

A prototype C++ microsimulation package inspired by [openm++](https://ompp.sourceforge.io/) and MODGEN. Models are defined in high-level code and executed in an embedded simulation framework written in C++. 

- Speed and flexibility are key requirements.

Currently evaluating embedded python. End goal is to produce a dynamic microsimulation of population in age, gender, ethnicity and location, using ethnicity-specific ASFR, ASMR and migration data. 

- First use case is a single local authority in a single thread.
- Second use case is an entire country using MPI.





