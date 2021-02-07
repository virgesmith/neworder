
import numpy as np
import neworder as no

from utils import assert_throws

def test_logistic_logit():

  n = 100 # wont work if odd!

  x = np.linspace(-10.0,10.0,n+1)
  y = no.stats.logistic(x)
  assert np.all(y >= -1)
  assert np.all(y <= 1)
  assert y[n//2] == 0.5

  assert np.all(np.fabs(y + y[::-1] - 1.0) < 1e-15)

  x2 = no.stats.logit(y)

  assert np.all(np.fabs(x2 - x) < 2e-12)