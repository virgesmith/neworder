"""
Membership

The model is a toy scheme membership: members join each year and
churn (leave) at a fixed annual hazard. finalise() then proves, empirically,
that SplitMix64 draws are stable under sub-sampling and reordering, and
contrasts this with the sequential MonteCarlo stream, which is not.
"""

from membership import Membership  # ty:ignore[unresolved-import]

import neworder as no

# Initial member population
N0 = 1000
p_churn = 0.1  # probability of leaving in a given year
p_join = 0.12  # probability of joining in a given year

timeline = no.LinearTimeline(2026, 2036, 10)
model = Membership(timeline, n0=N0, p_churn=p_churn, p_join=p_join)
ok = no.run(model)
if not ok:
    no.log("model failed!")
