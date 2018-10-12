
import numpy as np

# Dynamics parameters from https://github.com/virgesmith/ompp/blob/master/models/RiskPaths/parameters/Default/RiskPaths.dat
# //LABEL(RiskPaths) RiskPaths
# parameters {
# 	logical	CanDie = FALSE; // Switch to turn mortality on

# 	 // Age-specific death probabilities
# 	double	ProbMort[LIFE] = {
# 		(100) .01, 1, 
# 	};

mortality_rate = np.full(100, 0.01)
mortality_rate[-1] = 1.0

# 	 // Age baseline for first pregnancy
# 	double	AgeBaselinePreg1[AGEINT_STATE] = {
# 		0, 0.2869, 0.7591, 0.8458, 0.8167, 0.6727, 0.5105, 0.4882, 0.2562, 0.2597, 0.1542, 0, 
# 	};

#  f(AgeintState)
p_preg = np.ndarray([0.0, 0.2869, 0.7591, 0.8458, 0.8167, 0.6727, 0.5105, 0.4882, 0.2562, 0.2597, 0.1542, 0.0])

# 	 // Age baseline for 1st union formation
# 	double	AgeBaselineForm1[AGEINT_STATE] = {
# 		0, 0.030898, 0.134066, 0.167197, 0.165551, 0.147390, 0.108470, 0.080378, 0.033944, 0.045454, 0.040038, 0, 
# 	};

p_u1f = np.ndarray([0, 0.030898, 0.134066, 0.167197, 0.165551, 0.147390, 0.108470, 0.080378, 0.033944, 0.045454, 0.040038, 0])

# 	 // Relative risks of union status on first pregnancy
# 	double	UnionStatusPreg1[UNION_STATE] = {
# 		0.0648, 1.0000, 0.2523, 0.0648, 0.8048, 0.0648, 
# 	};
# See UnionState
#                     sgl      u1a     u1b     sgl     u2      sgl
r_preg = np.ndarray([0.0648, 1.0000, 0.2523, 0.0648, 0.8048, 0.0648])

# 	 // Separation Duration Baseline of 2nd Formation
# 	double	SeparationDurationBaseline[DISSOLUTION_DURATION] = {
# 		0.1995702, 0.1353028, 0.1099149, 0.0261186, 0.0456905, 
# 	};

# TODO what is this?
r_diss = np.ndarray([0.1995702, 0.1353028, 0.1099149, 0.0261186, 0.0456905])

# 	 // Union Duration Baseline of Dissolution
# 	double	UnionDurationBaseline[UNION_ORDER][UNION_DURATION] = {
# 		0.0096017, (2) 0.0199994, 0.0213172, 0.0150836, 0.0110791, 
# 		(2) 0.0370541, (2) 0.012775, (2) 0.0661157, 
# 	};
# };

# TODO what is this?
r_diss2 = np.ndarray([[0.0096017, 0.0199994, 0.0199994, 0.0213172, 0.0150836, 0.0110791],
                     [0.0370541, 0.0370541, 0.012775, 0.012775, 0.0661157, 0.0661157]])