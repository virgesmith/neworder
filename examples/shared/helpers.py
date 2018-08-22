"""
Common helper functionality
"""

import numpy as np

def check(data):
  # check no duplicated PID
  if len(data[data.duplicated(['PID'], keep=False)].head()):
    raise ValueError("Duplicate PIDs found")
  # Valid ETH, SEX, AGE
  if not np.array_equal(sorted(data.DC1117EW_C_SEX.unique()), [1,2]):
    raise ValueError("invalid gender value")
  if min(data.DC1117EW_C_AGE.unique().astype(int)) < 1 or \
     max(data.DC1117EW_C_AGE.unique().astype(int)) > 86:
    raise ValueError("invalid catgorical age value")
  # this can go below zero for cat 86+
  if (data.DC1117EW_C_AGE - data.Age).max() >= 1.0:
    raise ValueError("invalid fractional age value")

# # for testing
# if __name__ == "__main__":
#   raw_data = pd.read_csv("./NewETHPOP_inmig.csv")
#   lad="E08000021"
#   asir = create_from_ethpop(raw_data, lad)

#   print(asir.head())
