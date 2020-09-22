# Population Microsimulation

## Overview

In this example, the input data is a csv file containing a microsynthesised 2011 population of Newcastle generated from UK census data, by age, gender and ethnicity. The transitions modelled are: ageing, births, deaths and migrations, over a period of 40 years to 2051.

Births, deaths and migrations are are modelled using Monte-Carlo sampling (modelling a Poisson process) of distributions parameterised by age, sex and ethnicity-specific fertility, mortality and migration rates respectively, which are drawn from the NewETHPOP[[1]](#references.md) project.

For the fertility model newborns simply inherit their mother's location and ethnicity, are born aged zero, and have a randomly selected gender (even probability). The migration model is an 'in-out' model, i.e. it is not a full origin-destination model. Flows are either inward from 'elsewhere' or outward to 'elsewhere'.

People who have died are simply removed from the simulation.

Domestic migrations are given by rates per age, sex and ethnicity. Inward migration is based on the population ex-LAD, whereas outward migration is based on the population of the LAD being simulated.

International migrations are absolute (fractional) counts of individuals by age, sex and ethnicity, based on 2011 data. The values are rounded using a total-preserving algorithm. For emigration this presents a compilation: a situation can arise where a person who doesn't actually exist in the population is marked for migration.

Outward migrations are simply removed from the population. (They are not distributed in this model)

NB dealing with competing transitions...

During the simulation, at each timestep the check code computes and displays some summary data:

- the time
- the size of the population
- the mean age of the population
- the percentage of the population that are female
- the in and out migration numbers

At each checkpoint, the current population is written to a csv file.

{{ include_snippet("./docs/examples/src.md", show_filename=False) }}

## Setup

{{ include_snippet("examples/people/model.py") }}

## Model Implementation

population.py:

{{ include_snippet("examples/people/population.py") }}

## Execution

The model requires code in the shared directory, so this needs to be added to `PYTHONPATH`:

```bash
PYTHONPATH=examples/shared python examples/people/model.py
```

## Output

```text
...
[py 0/1]  check OK: time=2044.000 size=399199 mean_age=39.73, pct_female=48.41 net_migration=12181.0 (45857.0-33676.0)
[py 0/1]  check OK: time=2045.000 size=412334 mean_age=39.82, pct_female=48.43 net_migration=12824.0 (48139.0-35315.0)
[py 0/1]  check OK: time=2046.000 size=426426 mean_age=39.93, pct_female=48.49 net_migration=13954.0 (50436.0-36482.0)
[py 0/1]  check OK: time=2047.000 size=442034 mean_age=40.06, pct_female=48.65 net_migration=15587.0 (53304.0-37717.0)
[py 0/1]  check OK: time=2048.000 size=458712 mean_age=40.18, pct_female=48.81 net_migration=16712.0 (56205.0-39493.0)
[py 0/1]  check OK: time=2049.000 size=477050 mean_age=40.37, pct_female=48.96 net_migration=18540.0 (58914.0-40374.0)
[py 0/1]  check OK: time=2050.000 size=496943 mean_age=40.59, pct_female=49.10 net_migration=20368.0 (62157.0-41789.0)
[py 0/1]  check OK: time=2051.000 size=518993 mean_age=40.86, pct_female=49.30 net_migration=22545.0 (65413.0-42868.0)
[py 0/1]  writing ./examples/people/output/dm_E08000021_2051.000.csv
[py 0/1]  run time = 21.05s
```

This 40 year simulation of an initial population of about 280,000 growing to over half a million (no exogenous constraints) executed in about 20s on a single core of a medium-spec machine.
