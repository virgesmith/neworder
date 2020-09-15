
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
[py 0/1]  check OK: time=2012.000 size=288296 mean_age=37.25, pct_female=49.84 net_migration=5193.0 (27378.0-23545.0+2791.0-1431.0)
[py 0/1]  check OK: time=2013.000 size=294194 mean_age=37.15, pct_female=49.80 net_migration=4270.0 (27932.0-25172.0+2935.0-1425.0)
[py 0/1]  check OK: time=2014.000 size=297005 mean_age=37.25, pct_female=49.59 net_migration=1173.0 (27051.0-27431.0+3057.0-1504.0)
[py 0/1]  check OK: time=2015.000 size=297301 mean_age=37.47, pct_female=49.37 net_migration=-1271.0 (27130.0-29819.0+2938.0-1520.0)
[py 0/1]  check OK: time=2016.000 size=298170 mean_age=37.64, pct_female=49.39 net_migration=-575.0 (27239.0-29283.0+2937.0-1468.0)
[py 0/1]  check OK: time=2017.000 size=299832 mean_age=37.80, pct_female=49.30 net_migration=115.0 (26318.0-27709.0+2989.0-1483.0)
[py 0/1]  check OK: time=2018.000 size=301725 mean_age=37.94, pct_female=49.23 net_migration=499.0 (25950.0-26920.0+2999.0-1530.0)
[py 0/1]  writing ./examples/people/dm_E08000021_2018.000.csv
[py 0/1]  check OK: time=2019.000 size=303383 mean_age=38.07, pct_female=49.18 net_migration=283.0 (25668.0-26828.0+3014.0-1571.0)
[py 0/1]  check OK: time=2020.000 size=305316 mean_age=38.17, pct_female=49.06 net_migration=453.0 (25754.0-26754.0+3008.0-1555.0)
[py 0/1]  check OK: time=2021.000 size=307537 mean_age=38.29, pct_female=48.94 net_migration=810.0 (25436.0-26162.0+3070.0-1534.0)
[py 0/1]  writing ./examples/people/dm_E08000021_2021.000.csv
```

This 40 year simulation of an initial population of about 280,000 growing to over half a million (no exogenous constraints) executed in about 15s on a single core of a medium-spec machine.
