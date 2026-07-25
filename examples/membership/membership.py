"""
Membership
Showcases neworder's hash-based random stream. Unlike the sequential
MonteCarlo engine, SplitMix64 has no state to advance: the draw for a given
key (e.g. a person id) depends only on that key, never on how many other
draws preceded it. This makes it well suited to *open* populations, where
individuals join and leave over time and per-individual reproducibility
must survive changes in population size, membership or row order.
"""

import numpy as np
import pandas as pd

import neworder as no


# !constructor!
class Membership(no.Model):
    """
    An open population of scheme members. Each year some members churn
    (leave) and new members join. Demonstrates neworder.SplitMix64:
    - construct
    - step
    - finalise
    """

    def __init__(self, timeline: no.Timeline, n0: int, p_churn: float, p_join: float) -> None:
        super().__init__(timeline, no.MonteCarlo.deterministic_identical_stream)

        # Stateless (use_counter=False, the default): calling uarray() twice
        # with the same arguments always returns the same values. That's
        # exactly what's needed here - the churn draw for member i in year y
        # must be fixed, however many other members join or leave around it.

        # a stable master seed for this stochastic process - will be the same for each
        # draw from uarray(...) - identical arguments to this function will produce
        # identical results
        CHURN = no.SplitMix64.hash64("churn")
        self.rng = no.SplitMix64(lambda: CHURN)

        self.p_churn = p_churn
        self.avg_joiners = round(n0 * p_join)

        self.members = pd.DataFrame(index=no.df.unique_index(n0), data={"joined": timeline.start})
        self.members.index.name = "id"

        self.history: list[tuple[float, int]] = [(timeline.start, len(self.members))]

    # !constructor!

    # !step!
    def step(self) -> None:
        year = self.timeline.time

        # One SplitMix64 draw per member *id*, keyed on the process and the
        # year - not on a member's position in self.members, or on how many
        # other members are currently present.
        ids = self.members.index.to_numpy()
        u = self.rng.uarray(ids, int(year))
        self.members = self.members.loc[u >= self.p_churn]

        # New members joining are genuinely new events with no prior identity
        # to preserve, so an ordinary sequential draw is fine here.
        n_new = int(self.mc.ustream(1)[0] * 2 * self.avg_joiners)
        new_ids = no.df.unique_index(n_new)
        self.members = pd.concat([self.members, pd.DataFrame(index=new_ids, data={"joined": year})])
        self.members.index.name = "id"

        self.history.append((year + self.timeline.dt, len(self.members)))

    # !step!

    # !finalise!
    def finalise(self) -> None:
        for year, n in self.history:
            no.log(f"{year:.0f}: {n} members")

        self.prove_invariance()

    # !finalise!

    # !invariance!
    def prove_invariance(self) -> None:
        """
        The property that makes SplitMix64 suited to open populations: a
        member's draw depends only on its id, the process and the year -
        never on population size, membership or row order.
        """
        ids = self.members.index.to_numpy()
        year = int(self.timeline.time)

        # 1. sub-population invariance: drawing for the whole population and
        #    drawing for a single member in isolation agree for that member.
        whole = self.rng.uarray(ids, year)
        lone = self.rng.uarray(ids[:1], year)
        assert whole[0] == lone[0]

        # 2. reordering invariance: shuffling the ids changes their position
        #    in the output array, but not the value attached to any id.
        order = np.random.default_rng(0).permutation(len(ids))
        shuffled = self.rng.uarray(ids[order], year)
        np.testing.assert_array_equal(shuffled[np.argsort(order)], whole)

        no.log("SplitMix64: a member's draw is unchanged by sub-sampling or reordering")

        # Contrast with the sequential MonteCarlo stream: the value at
        # position i depends on when it was drawn, not on which id occupies
        # that position, so the same id can get a different value purely
        # because of where it sits in the DataFrame.
        self.mc.reset()
        original = dict(zip(ids.tolist(), self.mc.ustream(len(ids)).tolist(), strict=True))
        self.mc.reset()
        reordered = dict(zip(ids[order].tolist(), self.mc.ustream(len(ids)).tolist(), strict=True))
        changed = sum(original[i] != reordered[i] for i in ids)
        no.log(f"MonteCarlo: reordering alone changed the draw for {changed}/{len(ids)} members")

    # !invariance!
