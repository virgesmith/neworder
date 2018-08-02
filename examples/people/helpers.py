"""
Helper functions for the people example
"""

import pandas as pd

def census_eth_to_newethpop_eth(data):
  """ Maps census categories (DC2101EW_C_ETHPUK11) to NewEthpop. Note this is a one-way mapping """
  eth_map = { 0: "INV",
              1: "INV", 
              2: "WBI",
              3: "WHO", 
              4: "WHO",
              5: "WHO", 
              6: "MIX",
              7: "MIX",
              8: "MIX",
              9: "MIX",
              10: "MIX",
              11: "INV",
              12: "IND",
              13: "PAK",
              14: "BAN",
              15: "CHI",
              16: "OAS",
              17: "INV",
              18: "BLA",
              19: "BLC",
              20: "OBL",
              21: "OTH",
              22: "OTH", 
              23: "OTH" } 
  data["NewEthpop_ETH"] = data.DC2101EW_C_ETHPUK11.map(eth_map) #, na_action=None)
  return data.drop("DC2101EW_C_ETHPUK11", axis=1)


def create_migration_table(raw_data, lad):
  """ Processes raw NewETHPOP in/out migration data into a LAD-specific table that can be used efficiently """

  # As it's nonsensical to aggregate rates, we simply use the 85+ rate for everybody over 84
  # A population-weighted mean would perhaps be more accurate but unlikely to significantly affect the results 
  remove = ['M85.86', 'M86.87', 'M87.88', 'M88.89', 'M89.90', 'M90.91',
  'M91.92', 'M92.93', 'M93.94', 'M94.95', 'M95.96', 'M96.97', 'M97.98', 'M98.99',
  'M99.100', 'M100.101p', 'F85.86', 'F86.87',
  'F87.88', 'F88.89', 'F89.90', 'F90.91', 'F91.92', 'F92.93', 'F93.94', 'F94.95',
  'F95.96', 'F96.97', 'F97.98', 'F98.99', 'F99.100', 'F100.101p']

  data = raw_data.drop(remove, axis=1)

  # Filter by our location and remove other unwanted columns
  # partial match so works with census-merged LADs 
  data = data[data["LAD.code"].str.contains(lad)].drop(['Unnamed: 0', 'LAD.name', 'LAD.code'], axis=1)

  # "Melt" the table (collapsing all the age-sex columns into a single column containing)
  data = data.melt(id_vars=['ETH.group'])

  # Create separate age and sex columns
  data["DC1117_C_SEX"] = data.apply(lambda row: 1 if row.variable[0] == "M" else 2, axis=1)
  data["DC1117_C_AGE"] = data.apply(lambda row: int(row.variable.split(".")[1]) + 1, axis=1)
  # Remove another unneeded column
  data.drop(["variable"], axis=1, inplace=True)
  # Rename for consistency and set multiindex
  data.rename({"ETH.group": "NewEthpop_ETH", "value": "Rate"}, axis="columns", inplace=True)
  data.set_index(["NewEthpop_ETH", "DC1117_C_SEX", "DC1117_C_AGE"], inplace=True)
  return data

# # for testing
# if __name__ == "__main__":
#   raw_data = pd.read_csv("./NewETHPOP_inmig.csv")
#   lad="E08000021"
#   asir = create_migration_table(raw_data, lad)

#   print(asir.head())
