from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.special  # type: ignore


@dataclass
class Market:
    """Market data"""

    spot: float  # underlying spot price
    rate: float  # risk-free interest rate
    divy: float  # (continuous) dividend yield
    vol: float  # stock volatility


@dataclass
class Option:
    """(European) option instrument data"""

    callput: Literal["Call", "Put"]
    strike: float
    expiry: float


def _norm_cdf(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the normal cumulatve density funtion"""
    return (1.0 + scipy.special.erf(x / np.sqrt(2.0))) / 2.0


def analytic_pv(option: Option, market: Market) -> float:
    """Compute Black-Scholes European option price"""
    S = market.spot
    K = option.strike
    r = market.rate
    q = market.divy
    T = option.expiry
    vol = market.vol

    srt = vol * np.sqrt(T)
    rqs2t = (r - q + 0.5 * vol * vol) * T
    d1 = (np.log(S / K) + rqs2t) / srt
    d2 = d1 - srt
    df = np.exp(-r * T)
    qf = np.exp(-q * T)

    if option.callput == "CALL":
        return S * qf * _norm_cdf(d1) - K * df * _norm_cdf(d2)
    else:
        return -S * df * _norm_cdf(-d1) + K * df * _norm_cdf(-d2)
