
import numpy as np
from enum import Enum

# classification UNION_STATE
class UnionState(Enum):
  NEVER_IN_UNION = 0
  FIRST_UNION_PERIOD1 = 1
  FIRST_UNION_PERIOD2 = 2
  AFTER_FIRST_UNION = 3
  SECOND_UNION = 4
  AFTER_SECOND_UNION = 5

class Parity(Enum):
  CHILDLESS = 0
  PREGNANT = 1


def partition(start, finish, step=1):
  """ Helper function to return an inclusive equal-spaced range, i.e. finish will be the last element """
  # ensure finish is always included
  return np.append(np .arange(start, finish, step), finish)

# Dynamics parameters 

# Age of Consent at which the fertility rates begin
min_age = 15.0
max_age = 100.0

# //LABEL(RiskPaths) RiskPaths
# parameters {
# 	logical	CanDie = FALSE; // Switch to turn mortality on

# 	 // Age-specific death probabilities
# 	double	ProbMort[LIFE] = {
# 		(100) .01, 1,
# 	};

# flat mortality up to age 100 in 1y intervals
mortality_rate = np.full(int(max_age), 0.01)
mortality_rate[-1] = 1.0
mortality_delta_t = 1.0

# fertility rates given in 2.5y chunks from 15 to 40 incl
fertility_delta_t = 2.5
AgeintState = partition(min_age, 40, fertility_delta_t)

# 	 // Age baseline for first pregnancy
# 	double	AgeBaselinePreg1[AGEINT_STATE] = {
# 		0, 0.2869, 0.7591, 0.8458, 0.8167, 0.6727, 0.5105, 0.4882, 0.2562, 0.2597, 0.1542, 0,
# 	};

#  f(AgeintState)
p_preg = np.array([0.0, 0.2869, 0.7591, 0.8458, 0.8167, 0.6727, 0.5105, 0.4882, 0.2562, 0.2597, 0.1542, 0.0])

# 	 // Age baseline for 1st union formation
# 	double	AgeBaselineForm1[AGEINT_STATE] = {
# 		0, 0.030898, 0.134066, 0.167197, 0.165551, 0.147390, 0.108470, 0.080378, 0.033944, 0.045454, 0.040038, 0,
# 	};

p_u1f = np.array([0.0, 0.030898, 0.134066, 0.167197, 0.165551, 0.147390, 0.108470, 0.080378, 0.033944, 0.045454, 0.040038, 0.0])

# union1 lasts at least 3 years
min_u1 = 3.0

# 	 // Relative risks of union status on first pregnancy
# 	double	UnionStatusPreg1[UNION_STATE] = {
# 		0.0648, 1.0000, 0.2523, 0.0648, 0.8048, 0.0648,
# 	};
# See UnionState
#                     sgl      u1a     u1b     sgl     u2      sgl
r_preg = np.array([0.0648, 1.0000, 0.2523, 0.0648, 0.8048, 0.0648])

# taking this to mean union2 formation (=length of post-union1 single period)
# 	 // Separation Duration Baseline of 2nd Formation
# 	double	SeparationDurationBaseline[DISSOLUTION_DURATION] = {
# 		0.1995702, 0.1353028, 0.1099149, 0.0261186, 0.0456905,
# 	};


# UnionDuration = [1, 3, 5, 9, 13]
# currently need to modify above to have equal spacing
union_delta_t = 2.0
#                         1          3          5          7          9         11         13
r_u2f = np.array([0.1995702, 0.1353028, 0.1099149, 0.1099149, 0.0261186, 0.0261186, 0.0456905])

# Something wrong here: more data than dims
# 	 // Union Duration Baseline of Dissolution
# 	double	UnionDurationBaseline[UNION_ORDER][UNION_DURATION] = {
# 		0.0096017, (2) 0.0199994, 0.0213172, 0.0150836, 0.0110791,
# 		(2) 0.0370541, (2) 0.012775, (2) 0.0661157,
# 	};
# };

#                         1           3          5          7          9         11           13
r_diss2 = np.array([[0.0096017, 0.0199994, 0.0199994, 0.0199994, 0.0213172, 0.0150836, 0.0110791],
                   [0.0370541, 0.0370541, 0.012775,   0.012775,   0.012775, 0.0661157, 0.0661157]])
