# Chapter 1

This example is based on the example in the introductory chapter of [*Microsimulation and Population Dynamics*](../references.md). This is a simple example - a basic cohort model using continuous-time case-based simulation of mortality (only) for a homogeneous population, using a constant mortality hazard rate. In other words, age at death is sampled from an exponential distribution

\[
p(y)=\lambda e^{-\lambda y}
\]

which has a mean, i.e. life expectancy, of \(\mu=1/\lambda\).

The `neworder` implementation is as direct a port of the MODGEN model, as far as possible.

## Input

Here's the code that sets up and runs the model:

{{ include_snippet("./examples/chapter1/model.py") }}

## Implementation

Each individual is an instance of the `Person` class:

{{ include_snippet("./examples/chapter1/person.py", "person") }}

And the `People` model contains an array of `Person`s

{{ include_snippet("./examples/chapter1/person.py", "constructor") }}

The single timestep records each person's time of death

{{ include_snippet("./examples/chapter1/person.py", "step") }}

And the single checkpoint compares the mean of the sampled times of death with the expected value:

{{ include_snippet("./examples/chapter1/person.py", "checkpoint") }}

Finally this function is called from the model script when it displays the proportion of the cohort that are still alive at 10-year intervals:

{{ include_snippet("./examples/chapter1/person.py", "alive") }}

## Output

Some example output:

```text
[py 0/1] created 100000 individuals
[py 0/1] Life expectancy = 71.60 years (sampling error=0.17 years)
[py 0/1] Age 10 survival rate = 86.9%
[py 0/1] Age 20 survival rate = 75.6%
[py 0/1] Age 30 survival rate = 65.8%
[py 0/1] Age 40 survival rate = 57.3%
[py 0/1] Age 50 survival rate = 49.8%
[py 0/1] Age 60 survival rate = 43.3%
[py 0/1] Age 70 survival rate = 37.5%
[py 0/1] Age 80 survival rate = 32.7%
[py 0/1] Age 90 survival rate = 28.3%
[py 0/1] Age 100 survival rate = 24.7%
```
And with a constant mortality rate we see time of death heavily skewed towards younger ages than would be expected.

In the `neworder` framework, a more natural (and efficient) implementation would not use a class instance to represent an individual, but rather use a pandas DataFrame to store the population, with each row representing an individual, allowing bulk operations on the entire population. This approach is taken in some of the more complex examples, see for example [mortality](./mortality.md).

## [Examples Source Code](https://github.com/virgesmith/neworder/tree/master/examples)