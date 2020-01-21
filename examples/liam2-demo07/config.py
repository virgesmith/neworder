""" 
port of demo07 model from LIAM2 
see http://liam2.plan.be/
"""

# HDF5 support (in pandas) requires pytables, on ubuntu:
# sudo apt install python3-tables

from pathlib import Path
import neworder

neworder.timeline = (2016, 2018, 2)

neworder.data_dir = Path("../LIAM2/examples")

# input:
#     file: demo.h5
input_data = neworder.data_dir / "demo.h5"

# output:
#     path: output
#     file: simulation.h5
output_file = neworder.data_dir / "simulation.h5"


neworder.do_checks = False
neworder.log_level = 1

# init:
# - household: [init_reports, csv_output]
# - person: [init_reports, csv_output]
neworder.initialisations = {
  "retirement_age": { "module": "people", "class_": "RetirementAge", "args": (input_data, "/globals/periodic") },
  "people": { "module": "people", "class_": "People", "args": (input_data, "/entities/person") },
  "households": { "module": "households", "class_": "Households", "args": (input_data, "/entities/household") }
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
  "p_age": "people.ageing()",
  "p_death": "people.death()",
  "p_birth": "people.birth()",
  "p_divorce": "people.divorce()",
  "p_moveout": "people.get_a_life(households)",
  "h_summary" : "households.csv_output()",
  "h_prune" : "households.clean_empty(people)"
  #"test": "log(people.active_age(20))"
}

neworder.checkpoints = {
  "h_summary" : "households.write_reports('hh_size.csv')",
  "p_output" : "people.csv_output()",
}

# default_entity: person
