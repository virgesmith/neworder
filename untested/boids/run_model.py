import neworder as no

import matplotlib.pyplot as plt
from boids2d import Boids

N = 100

m = Boids(N)

no.log("q to quit")
no.run(m)
plt.show()


