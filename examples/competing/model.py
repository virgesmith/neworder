"""
Competing risks - fertility & mortality
"""
import numpy as np
import neworder

from people import People


# This is case-based model so no timeline is required...
# ... but we do need a delta-t between entries in fertility/mortality data
timeline = neworder.Timeline.null()

# Choose a simple linearly increasing mortality rate: 0.1% aged 0 to 2.5% aged 100
fertility_hazard_file = "examples/shared/NewETHPOP_fertility.csv"
mortality_hazard_file = "examples/shared/NewETHPOP_mortality.csv"
population_size = 100000
lad = "E09000030"
ethnicity = "WBI"

neworder.model = People(timeline, fertility_hazard_file, mortality_hazard_file, lad, ethnicity, population_size)

