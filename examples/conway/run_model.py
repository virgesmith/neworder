
from conway import Conway
import neworder as no

# size of domain
nx, ny = (480, 360)

# saturation (proportion initially alive)
sat = 0.36

n = int(nx * ny * sat)

# edges wrap - try with no.Domain.CONSTRAIN
m = Conway(nx, ny, n, no.Domain.WRAP)

no.run(m)
