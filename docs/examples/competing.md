# Competing Risks

This is a case-based continuous-time microsimulation of the competing risks of (multiple) fertility and mortality. The former is sampled using a nonhomogeneous multiple-arrival-time simulation of a Poisson process, with a minimum gap between events of 9 months. Mortality is sampled using a standard nonhomogeneous Poisson process. A mortality event before a birth event cancels the birth event.

The figure below shows the distribution of up to four births (stacked) plus mortality.

![Competing Fertility-Mortality histogram](./img/competing_hist_100k.png)
