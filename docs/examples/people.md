
# Polulation Microsimulation (Serial)

In this example, the input data is a csv file containing a microsynthesised 2011 population of Newcastle generated from UK census data, by age, gender and ethnicity. The transitions modelled are: ageing, births, deaths and migrations.

Ageing simply increments individual's ages according to the timestep.

Births, deaths and migrations are are modelled using Monte-Carlo sampling (modelling a Poisson process) of distributions parameterised by age, sex and ethnicity-specific fertility, mortality and migration rates respectively, which are drawn from the NewETHPOP[[1]](#references) project.

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

See [config.py](examples/people/config.py) for the simulation setup and [population.py](examples/people/population.py) for details of the model implementation.

The file [helpers.py](examples/people/helpers.py) defines some helper functions, primarily to reformat input data into a format that can be used efficiently.

```bash
$ time ./run_example.sh people
[no 0/1] env: seed=19937 python 3.6.6 (default, Sep 12 2018, 18:26:19)  [GCC 8.0.1 20180414 (experimental) [trunk revision 259383]]
[no 0/1] starting microsimulation...
[no 0/1] t=2011.250000 initialise: people
[no 0/1] t=2012.250000 transition: age
[no 0/1] t=2012.250000 transition: fertility
[no 0/1] t=2012.250000 transition: migration
[no 0/1] t=2012.250000 transition: mortality
[no 0/1] t=2012.250000 check: check
[py 0/1] check OK: time=2012.250 size=281728 mean_age=37.47, pct_female=49.84 net_migration=1409 (23626-23765+2927-1379)
...
[no 0/1] t=2049.250000 transition: age
[no 0/1] t=2049.250000 transition: fertility
[no 0/1] t=2049.250000 transition: migration
[no 0/1] t=2049.250000 transition: mortality
[no 0/1] t=2049.250000 check: check
[py 0/1] check OK: time=2049.250 size=566509 mean_age=40.16, pct_female=49.69 net_migration=27142 (70953-46534+5673-2950)
[no 0/1] t=2050.250000 transition: age
[no 0/1] t=2050.250000 transition: fertility
[no 0/1] t=2050.250000 transition: migration
[no 0/1] t=2050.250000 transition: mortality
[no 0/1] t=2050.250000 check: check
[py 0/1] check OK: time=2050.250 size=594350 mean_age=40.42, pct_female=49.94 net_migration=30095 (75464-48243+6003-3129)
[no 0/1] t=2050.250000 checkpoint: write_table
[py 0/1] writing ./examples/people/dm_E08000021_2050.250.csv
[no 0/1] SUCCESS
```

This 40 year simulation of a population of about 280,000 more than doubling (no exogenous constraints) executed in about 25s on a single core on a desktop machine.
