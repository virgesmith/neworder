
from math import erf, sqrt, log, exp

def norm_cdf(x):
  """ Compute the inverse normal cumulatve density funtion """
  return (1.0 + erf(x / sqrt(2.0))) / 2.0

def bs_euro_option(S, K, r, q, T, vol, callput):
  """ Compute Black-Schole European option price """
  srt = vol * sqrt(T)
  rqs2t = (r - q + 0.5 * vol * vol) * T
  d1 = (log(S/K) + rqs2t) / srt
  d2 = d1 - srt
  df = exp(-r * T)
  qf = exp(-q * T)

  if callput == "CALL":
    return S * qf * norm_cdf(d1) - K * df * norm_cdf(d2)
  else:
    return -S * df * norm_cdf(-d1) + K * df * norm_cdf(-d2)
