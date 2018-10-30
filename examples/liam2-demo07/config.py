""" 
port of demo07 model from LIAM2 
see http://liam2.plan.be/
"""

import neworder

neworder.timeline = (2016, 2018, 2)

# HDF5 requires pytables, on ubuntu:
# sudo apt install python3-tables


# input:
#     file: demo.h5
input_data = "../LIAM2/examples/demo.h5"

# output:
#     path: output
#     file: simulation.h5
output_file = "./simulation.h5"


neworder.do_checks = False
neworder.log_level = 1

# init:
#     - household: [init_reports, csv_output]
#     - person: [init_reports, csv_output]
neworder.initialisations = {
  "retirement_age": { "module": "people", "class_": "RetirementAge", "parameters": [input_data, "/globals/periodic"] },
  "people": { "module": "people", "class_": "People", "parameters": [input_data, "/entities/person"] },
  "households": { "module": "households", "class_": "Households", "parameters": [input_data, "/entities/household"] }
}

# processes:
# - person: [
#     ageing,
#     birth,
#     death,
# ]
# - household: [clean_empty]
# - person: [
#     marriage,
#     get_a_life,
#     divorce,
# ]
# # reporting
# - person: [
# #            chart_demography,
#     civilstate_changes,
#     csv_output
# ]
# - household: [csv_output]
neworder.transitions = {
}

neworder.checkpoints = {
  "save_data" : "log(retirement_age(2034))",
}

# default_entity: person
