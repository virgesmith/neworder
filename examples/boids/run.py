import sys
import neworder as no

# perfomance can be improved by **reducing** the number of threads numpy uses
# e.g. set OPENBLAS_NUM_THREADS=2
# not sure why but might be contention with the graphical rendering

N = 1000 # number of boids
range = 1.0 # extent of the domain
vision = 0.2 # distance boids "see"
exclusion = 0.05 # distance collision avoidance kicks in
speed = 1.0

if len(sys.argv) != 2 or sys.argv[1] not in ["2d", "3d"]:
  print("usage: python examples/boids/run.py 2d|3d")
  exit(1)

if sys.argv[1] == "2d":
  from boids2d import Boids2d as Boids
else:
  from boids3d import Boids3d as Boids

m = Boids(N, range, vision, exclusion, speed)

no.log("p to pause, q to quit")
no.run(m)
