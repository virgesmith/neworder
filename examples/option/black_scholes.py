"""Black-Scholes model implementations: analytic and MC"""

import numpy as np
from helpers import Market, Option, analytic_pv

import neworder


# Subclass neworder.Model
class BlackScholes(neworder.Model):
    # !constructor!
    def __init__(self, option: Option, market: Market, nsims: int) -> None:
        # Using exact MC calc of GBM requires only 1 timestep
        timeline = neworder.LinearTimeline(0.0, option.expiry, 1)
        super().__init__(timeline, neworder.MonteCarlo.deterministic_identical_stream)

        self.rng = neworder.as_np(self.mc)
        self.option = option
        self.market = market
        self.nsims = nsims

    # !constructor!

    # !modifier!
    def modify(self) -> None:
        if neworder.mpi.RANK == 1:
            self.market.spot *= 1.01  # delta/gamma up bump
        elif neworder.mpi.RANK == 2:
            self.market.spot *= 0.99  # delta/gamma down bump
        elif neworder.mpi.RANK == 3:
            self.market.vol += 0.001  # 10bp upward vega

    # !modifier!

    # !step!
    def step(self) -> None:
        self.pv = self.simulate()

    # !step!

    # !check!
    def check(self) -> bool:
        # check the rng streams are still in sync by sampling from each one,
        # comparing, and broadcasting the result. If one process fails the
        # check and exits without notifying the others, deadlocks can result.
        # send the state representation to process 0 (others will get None)
        states = neworder.mpi.COMM.gather(self.mc.state(), 0)
        # process 0 checks the values
        if states:
            ok = all(s == states[0] for s in states)
        else:
            ok = True
        # broadcast process 0's ok to all processes
        ok = neworder.mpi.COMM.bcast(ok, root=0)
        return ok

    # !check!

    # !finalise!
    def finalise(self) -> None:
        # check and report accuracy
        """Compare MC price to analytic"""
        ref = analytic_pv(self.option, self.market)
        err = self.pv / ref - 1.0
        neworder.log("mc: {:.6f} / ref: {:.6f} err={:.2%}".format(self.pv, ref, err))
        # relative error should be within O(1/(sqrt(sims))) of analytic solution
        if abs(err) > 2.0 / np.sqrt(self.nsims):
            neworder.log("MC error is larger than expected")

        # compute and report some market risk
        self.greeks()

    # !finalise!

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

    def greeks(self) -> None:
        # get all the results
        pvs = neworder.mpi.COMM.gather(self.pv, 0)
        # compute sensitivities on rank 0
        if pvs:
            neworder.log(f"PV={pvs[0]:.3f}")
            neworder.log(f"delta={(pvs[1] - pvs[2]) / 2:.3f}")
            neworder.log(f"gamma={(pvs[1] - 2 * pvs[0] + pvs[2]):.3f}")
            neworder.log(f"vega 10bp={pvs[3] - pvs[0]:.3f}")
