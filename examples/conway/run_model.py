
from conway import Conway
import neworder as no

# size of domain
nx, ny = (320, 320)

# saturation (proportion initially alive)
sat = 0.36

n = int(nx * ny * sat)

# edges wrap - try with no.Edge.CONSTRAIN
m = Conway(nx, ny, n, no.Edge.WRAP)

no.run(m)
