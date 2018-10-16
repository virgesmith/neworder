
import numpy as np

# Dynamics parameters from https://github.com/virgesmith/ompp/blob/master/models/RiskPaths/parameters/Default/RiskPaths.dat
# //LABEL(RiskPaths) RiskPaths
# parameters {
# 	logical	CanDie = FALSE; // Switch to turn mortality on

# 	 // Age-specific death probabilities
# 	double	ProbMort[LIFE] = {
# 		(100) .01, 1, 
# 	};

max_age = 100.0

mortality_rate = np.full(int(max_age), 0.01)
mortality_rate[-1] = 1.0

# Age of Consent at which the rates begin 
min_age = 15.0

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

delta_t = 2.5
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


# UNION_DURATION is defined as 1,3,5,9,13
# currently need to modify to have equal spacing

delta_t_u = 2.0
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
