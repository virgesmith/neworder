"""
Black-Scholes model implementations: analytic and MC
Using multithreading rather than MPI
"""

from copy import copy

import numpy as np
from helpers import Market, Option, analytic_pv

import neworder


# Subclass neworder.Model
class BlackScholesMT(neworder.Model):
    """Multithreaded variant"""

    def __init__(self, option: Option, market: Market, nsims: int, index: int) -> None:
        # Using exact MC calc of GBM requires only 1 timestep
        timeline = neworder.LinearTimeline(0.0, option.expiry, 1)
        # ensures all threads have identical random streams
        super().__init__(timeline, neworder.MonteCarlo.deterministic_independent_stream)

        self.rng = neworder.as_np(self.mc)
        self.option = option
        self.market = copy(market)  # copy as will be modifying (not necessary with MPI)
        self.nsims = nsims
        # safer to explicitly set the index rather than relying on potentially nondeterministic thread_id
        self.index = index

    def modify(self) -> None:
        """Alter the market data depending on which thread we are"""
        if self.index == 1:
            self.market.spot *= 1.01  # delta/gamma up bump
        elif self.index == 2:
            self.market.spot *= 0.99  # delta/gamma down bump
        elif self.index == 3:
            self.market.vol += 0.001  # 10bp upward vega

    def step(self) -> None:
        self.pv = self.simulate()

    def check(self) -> bool:
        # Its not straightforward for one model thread to compare its RNG state to the other thread (c.f. MPI
        # implementation). Solution is to defer the comparisons to the main thread.
        return True

    def finalise(self) -> None:
        """Compare MC price to analytic"""
        ref = analytic_pv(self.option, self.market)
        err = self.pv / ref - 1.0
        neworder.log(f"mc: {self.pv:.6f} / ref: {ref:.6f} err={err:.2%}")
        # relative error should be within O(1/(sqrt(sims))) of analytic solution
        if abs(err) > 2.0 / np.sqrt(self.nsims):
            neworder.log("MC error is larger than expected")

    def simulate(self) -> float:
        # get the single timestep from the timeline
        dt = self.timeline.dt
        normals = self.rng.normal(size=self.nsims)

        # compute underlying prices at t=dt
        S = self.market.spot
        r = self.market.rate
        q = self.market.divy
        sigma = self.market.vol
        underlyings = S * np.exp((r - q - 0.5 * sigma * sigma) * dt + normals * sigma * np.sqrt(dt))
        # compute option prices at t=dt
        if self.option.callput == "CALL":
            fv = (underlyings - self.option.strike).clip(min=0.0).mean()
        else:
            fv = (self.option.strike - underlyings).clip(min=0.0).mean()

        # discount back to val date
        return fv * np.exp(-r * dt)


# this is outside the model as it called by the main thread after all the model threads have completed
def greeks(models: list[BlackScholesMT]) -> None:
    # get all the results
    pvs = {m.index: m.pv for m in models}
    # compute sensitivities on rank 0
    neworder.log(f"PV={pvs[0]:.3f}")
    neworder.log(f"delta={(pvs[1] - pvs[2]) / 2:.3f}")
    neworder.log(f"gamma={(pvs[1] - 2 * pvs[0] + pvs[2]):.3f}")
    neworder.log(f"vega 10bp={pvs[3] - pvs[0]:.3f}")
