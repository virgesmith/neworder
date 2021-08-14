
from conway import Conway
import neworder as no

# size of domain
nx, ny = (640, 480)

# saturation (proportion initially alive)
sat = 0.36

n = int(nx * ny * sat)

# edges do not wrap - try with no.Domain.WRAP
m = Conway(nx, ny, n, no.Domain.CONSTRAIN)

no.run(m)
