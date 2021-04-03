import neworder as no

import matplotlib.pyplot as plt
from boids2d import Boids

N = 100

m = Boids(N)

no.log("<space> to pause, q to quit, r to resume")
no.run(m)
plt.show()


