# MODGEN examples

The examples here are based on the examples given in the book [*Microsimulation and Population Dynamics*](../../README.md#references). 

## Chapter 1
This is the first and simplest example - a basic cohort model using continuous-time case-based simulation of mortality (only) for a homogeneous population, using a constant mortality hazard rater. The `neworder` implementation is as direct a port of the model as possible. 

The model configuration is [here](../../examples/Chapter_1/config.py) and the implementation [here](../../examples/Chapter_1/person.py)

Some example output:
```
$ ./run_example.sh Chapter_1
[no 0/1] env: seed=19937 python 3.6.6 (default, Sep 12 2018, 18:26:19)  [GCC 8.0.1 20180414 (experimental) [trunk revision 259383]]
[no 0/1] starting microsimulation. timestep=0.000000, checkpoint(s) at [1]
[no 0/1] t=0.000000(0) initialise: people
[py 0/1] created 100000 individuals
[no 0/1] t=0.000000(1) transition: time_of_death
[no 0/1] t=0.000000(1) checkpoint: life_expectancy
[py 0/1] Life expectancy = 71.13 years (sampling error=-0.300793)
[no 0/1] SUCCESS exec time=0.150225s
```

## Mortality

This is based on the example in the second chapter of the book - *The Life Table*. It uses an age-specific mortality rate and has two model implementations - a direct MODGEN port: [person.py](../../examples/mortality/person.py), and a re-implementation taking advantage of python packages: [people.py](../../examples/mortality/people.py). The latter also includes a visualisation of the results:

[Mortality](../../examples/mortality/config.py). 

![Mortality histogram - 10000 people](./img/mortality_hist_10k.gif) ![Mortality histogram - 100000 people](./img/mortality_hist_100k.gif)

The mortality data is sourced from the NewETHPOP project (TODO ref) and represents the mortality rate for white British males in one of the London Boroughs.

The "pythonic" implementation uses a pandas dataframe to store the population, as opposed to an array of `Person` objects representing each individual. This a struct-of-arrays rather than array-of-structs approach is, in this instance, far more efficient, running roughly three times quicker. It retains the basic algorithm: 
- each year, sample time of death for alive individuals 
- if year is not at the end of the mortality table
  - if death occurs within the year, record time of death mark individual as dead
  - otherwise, increment age by 1 and resample  
- else
  - record time of death mark individual as dead
- take mean time of death

An even more efficient implementation is to sample the time of death directly using the Lewis-Shedler [[4]](../../README.md#references) "thinning" algorithm - this approach doesn't require a time loop.

