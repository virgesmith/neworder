"""
Black-Scholes model implementations: analytic and MC
Using multithreading rather than MPI
"""

from typing import Any

import numpy as np
from helpers import norm_cdf, nstream

import neworder


# Subclass neworder.Model
class BlackScholesMT(neworder.Model):
    def __init__(self, option: dict[str, Any], market: dict[str, float], nsims: int, index: int) -> None:
        # Using exact MC calc of GBM requires only 1 timestep
        timeline = neworder.LinearTimeline(0.0, option["expiry"], 1)
        super().__init__(timeline, neworder.MonteCarlo.deterministic_independent_stream)

        self.option = option
        self.market = market.copy()  # as will be modifying
        self.nsims = nsims
        # safer to explicitly set the index rather than relying on potentially nondeterministic thread_index
        self.index = index

    def modify(self) -> None:
        if self.index == 1:
            self.market["spot"] *= 1.01  # delta/gamma up bump
        elif self.index == 2:
            self.market["spot"] *= 0.99  # delta/gamma down bump
        elif self.index == 3:
            self.market["vol"] += 0.001  # 10bp upward vega

    def step(self) -> None:
        self.pv = self.simulate()

    def finalise(self) -> None:
        # check and report accuracy
        self.compare()

    def simulate(self) -> float:
        # get the single timestep from the timeline
        dt = self.timeline.dt
        normals = nstream(self.mc.ustream(self.nsims))

        # compute underlying prices at t=dt
        S = self.market["spot"]
        r = self.market["rate"]
        q = self.market["divy"]
        sigma = self.market["vol"]
        underlyings = S * np.exp((r - q - 0.5 * sigma * sigma) * dt + normals * sigma * np.sqrt(dt))
        # compute option prices at t=dt
        if self.option["callput"] == "CALL":
            fv = (underlyings - self.option["strike"]).clip(min=0.0).mean()
        else:
            fv = (self.option["strike"] - underlyings).clip(min=0.0).mean()

        # discount back to val date
        return fv * np.exp(-r * dt)

    def analytic(self) -> float:
        """Compute Black-Scholes European option price"""
        S = self.market["spot"]
        K = self.option["strike"]
        r = self.market["rate"]
        q = self.market["divy"]
        T = self.option["expiry"]
        vol = self.market["vol"]

        srt = vol * np.sqrt(T)
        rqs2t = (r - q + 0.5 * vol * vol) * T
        d1 = (np.log(S / K) + rqs2t) / srt
        d2 = d1 - srt
        df = np.exp(-r * T)
        qf = np.exp(-q * T)

        if self.option["callput"] == "CALL":
            return S * qf * norm_cdf(d1) - K * df * norm_cdf(d2)
        else:
            return -S * df * norm_cdf(-d1) + K * df * norm_cdf(-d2)

    def compare(self) -> bool:
        """Compare MC price to analytic"""
        ref = self.analytic()
        err = self.pv / ref - 1.0
        neworder.log(f"mc: {self.pv:.6f} / ref: {ref:.6f} err={err:.2%}")
        # relative error should be within O(1/(sqrt(sims))) of analytic solution
        return True if abs(err) <= 2.0 / np.sqrt(self.nsims) else False


def greeks(models: list[BlackScholesMT]) -> None:
    # get all the results
    pvs = {m.index: m.pv for m in models}
    # compute sensitivities on rank 0
    neworder.log(f"PV={pvs[0]:.3f}")
    neworder.log(f"delta={(pvs[1] - pvs[2]) / 2:.3f}")
    neworder.log(f"gamma={(pvs[1] - 2 * pvs[0] + pvs[2]):.3f}")
    neworder.log(f"vega 10bp={pvs[3] - pvs[0]:.3f}")
