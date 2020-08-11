"""
Helper functions for formatting NewETHPOP fetility, mortality and migration data
"""

import pandas as pd
import numpy as np
#import humanleague as hl

UK_POP_2011 = 62740000.0
ALL_CATEGORIES = 2 * 86 * 12 # gender by age by eth

def from_census_eth(data):
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

def update_lad_codes(data):
  """ Updates LAD codes to 2015 """
  lad_mapping = { "E08000020":  "E08000037", # Gateshead
                  "E06000048":  "E06000057", # Northumberland
                  "E07000100":  "E07000240", # St Albans
                  "E07000101":  "E07000243", # Stevenage
                  "E07000104":  "E07000241", # Welwyn-Hatfield
                  "E07000097":  "E07000242", # East Hertfordshire
  }

  data.replace(lad_mapping, inplace=True) #[colname]
  return data


def local_rate_from_national_rate(data, localpop):
  """
  Scales up a rate based on national to that of local
  """
  # 2011 UK population was ~62.74M
  scale = UK_POP_2011 / localpop
  data.Rate = data.Rate * scale
  return data

def local_rates_from_national_rate(data, pop):
  """
  Multi-lad version of above
  """
  lads = pop.LAD.unique()
  #print(lads)
  #print(pop.head())
  #print(data.head())

  for lad in lads:
    localpop = len(pop[pop.LAD == lad])
    scale = UK_POP_2011 / localpop
    # deal with census merged LADs
    if lad == "E09000001" or lad == "E09000033":
      scale = UK_POP_2011 / (7397 + 219340)
    elif lad == "E06000052" or lad == "E06000053":
      scale = UK_POP_2011 / (533514 + 2222)

    # TODO fix PerformanceWarning: indexing past lexsort depth may impact performance.
    data.loc[lad, "Rate"] = data.loc[lad, "Rate"].values * scale

  return data

def local_rate_rescale_from_absolute(data, localpop):
  """
  Turns absolute (2011) values into rates. Use for intl migration figures that are absolute values per LAD
  """
  # This is roughly rescaled by age(86)/gender(2)/eth(12)
  scale = ALL_CATEGORIES / localpop
  data.Rate = data.Rate * scale
  return data

def local_rates_from_absolute(data, pop):
  """
  Multi-lad version of above
  """
  lads = pop.LAD.unique()

  for lad in lads:
    localpop = len(pop[pop.LAD == lad])
    scale = ALL_CATEGORIES / localpop
    # deal with census merged LADs
    if lad == "E09000001" or lad == "E09000033":
      scale = ALL_CATEGORIES / (7397 + 219340)
    elif lad == "E06000052" or lad == "E06000053":
      scale = ALL_CATEGORIES / (533514 + 2222)

    #print(lad, scale)
    # TODO fix PerformanceWarning: indexing past lexsort depth may impact performance.
    data.loc[lad, "Rate"] = data.loc[lad, "Rate"].values * scale

  return data

def create(raw_data, lad, truncate85=True):
  """ Processes raw NewETHPOP in/out migration data into a LAD-specific table that can be used efficiently """

  # Truncate to census (2011) max age of 85+
  if truncate85:
  # As it's nonsensical to aggregate rates, we simply use the 85+ rate for everybody over 84
  # A population-weighted mean would perhaps be more accurate but unlikely to significantly affect the results
    remove = ['M85.86', 'M86.87', 'M87.88', 'M88.89', 'M89.90', 'M90.91',
    'M91.92', 'M92.93', 'M93.94', 'M94.95', 'M95.96', 'M96.97', 'M97.98', 'M98.99',
    'M99.100', 'M100.101p', 'F85.86', 'F86.87',
    'F87.88', 'F88.89', 'F89.90', 'F90.91', 'F91.92', 'F92.93', 'F93.94', 'F94.95',
    'F95.96', 'F96.97', 'F97.98', 'F98.99', 'F99.100', 'F100.101p']
    data = update_lad_codes(raw_data.drop(remove, axis=1))
  else:
    rename = {'M100.101p': 'M100.101', 'F100.101p': 'F100.101'}
    data = update_lad_codes(raw_data.rename(columns=rename))

  # Filter by our location and remove other unwanted columns
  # partial match so works with census-merged LADs
  data = data[data["LAD.code"].str.contains(lad)].drop(['Unnamed: 0', 'LAD.name', 'LAD.code'], axis=1)

  # "Melt" the table (collapsing all the age-sex columns into a single column containing)
  data = data.melt(id_vars=['ETH.group'])

  # Create separate age and sex columns
  data["DC1117EW_C_SEX"] = data.apply(lambda row: 1 if row.variable[0] == "M" else 2, axis=1)
  data["DC1117EW_C_AGE"] = data.apply(lambda row: int(row.variable.split(".")[1]) + 1, axis=1)
  # Remove another unneeded column
  data.drop(["variable"], axis=1, inplace=True)
  # Rename for consistency and set multiindex
  data.rename({"ETH.group": "NewEthpop_ETH", "value": "Rate"}, axis="columns", inplace=True)
  data.set_index(["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"], inplace=True)
  return data

def create_multi(raw_data, lads):
  """ Processes raw NewETHPOP in/out migration data into a LAD-filtered table that can be used efficiently """

  # As it's nonsensical to aggregate rates, we simply use the 85+ rate for everybody over 84
  # A population-weighted mean would perhaps be more accurate but unlikely to significantly affect the results
  remove = ['M85.86', 'M86.87', 'M87.88', 'M88.89', 'M89.90', 'M90.91',
  'M91.92', 'M92.93', 'M93.94', 'M94.95', 'M95.96', 'M96.97', 'M97.98', 'M98.99',
  'M99.100', 'M100.101p', 'F85.86', 'F86.87',
  'F87.88', 'F88.89', 'F89.90', 'F90.91', 'F91.92', 'F92.93', 'F93.94', 'F94.95',
  'F95.96', 'F96.97', 'F97.98', 'F98.99', 'F99.100', 'F100.101p']

  data = update_lad_codes(raw_data.drop(remove, axis=1))

  # Fix census-merged LADs (doesn't play well with join on multiindex)
  # This also requires rescaling the values by the relative sizes of the LADs
  # e.g. 7397 / 219340 for City/Westminster
  # which is done in the local_ratestowards the end of this function
  data.replace("E09000001+E09000033", "E09000001", inplace=True)
  data.replace("E06000052+E06000053", "E06000052", inplace=True)
  dups = data[(data["LAD.code"] == "E09000001") | (data["LAD.code"] == "E06000052")].copy()
  dups.replace("E09000001", "E09000033", inplace=True)
  dups.replace("E06000052", "E06000053", inplace=True)
  data = data.append(dups)

  # Filter by our location and remove other unwanted columns
  # partial match so works with census-merged LADs
  data = data[data["LAD.code"].isin(lads)].drop(['Unnamed: 0', 'LAD.name'], axis=1)

  # "Melt" the table (collapsing all the age-sex columns into a single column containing the original column headings)
  data = data.melt(id_vars=['ETH.group', 'LAD.code'])

  # Create separate age and sex columns
  data["DC1117EW_C_SEX"] = data.apply(lambda row: 1 if row.variable[0] == "M" else 2, axis=1)
  data["DC1117EW_C_AGE"] = data.apply(lambda row: int(row.variable.split(".")[1]) + 1, axis=1)
  # Remove another unneeded column
  data.drop(["variable"], axis=1, inplace=True)
  # Rename for consistency and set multiindex
  data.rename({'LAD.code': "LAD", "ETH.group": "NewEthpop_ETH", "value": "Rate"}, axis="columns", inplace=True)

  data.set_index(["LAD", "NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"], inplace=True)
  return data

