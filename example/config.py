
# config.py

timespan = [2011, 2020]
timestep = 1

initial_population = "src/test/ssm_E09000001_MSOA11_ppp_2011.csv"

mortality_hazard = 0.01

transitions = {
#  "migration": ["Area"],
  "age": ["DC1117EW_C_AGE"],
#  "birth": [] 
#  "death": []
}
