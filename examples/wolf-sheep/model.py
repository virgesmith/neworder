
import neworder as no
from wolf_sheep import WolfSheep
import matplotlib.pyplot as plt

#no.verbose()

params = {
  "grid": {
    "width": 100,
    "height": 100
  },
  "wolves": {
    "starting_population": 150,
    "reproduce": 0.05,
    "speed": 2.4,
    "speed_variance": 0.01,
    "gain_from_food": 20
  },
  "sheep": {
    "starting_population": 300,
    "reproduce": 0.04,
    "speed": 0.9,
    "speed_variance": 0.01,
    "gain_from_food": 4
  },
  "grass": {
    "regrowth_time": 30
  }
}

m = WolfSheep(params)

# test cell assignment
# for _,r in m.wolves.iterrows():
#   c = int(r.cell)
#   print(r.x, r.y, c, m.grass.loc[c, "x"], m.grass.loc[c,"y"])
# for _,r in m.sheep.iterrows():
#   c = int(r.cell)
#   print(r.x, r.y, c, m.grass.loc[c, "x"], m.grass.loc[c,"y"])


no.run(m)


