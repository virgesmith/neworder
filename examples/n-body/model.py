
import neworder as no

from n_body import NBody

G = 0.01
N = 50
dt = 0.01

m = NBody(N, G, dt)

no.run(m)