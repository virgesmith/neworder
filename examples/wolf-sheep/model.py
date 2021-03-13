
import neworder as no
from wolf_sheep import WolfSheep
import matplotlib.pyplot as plt
width = 100
height = 100

n_wolves = 150
n_sheep = 300


m = WolfSheep(width, height, n_wolves, n_sheep)

# test cell assignment
# for _,r in m.wolves.iterrows():
#   c = int(r.cell)
#   print(r.x, r.y, c, m.grass.loc[c, "x"], m.grass.loc[c,"y"])
# for _,r in m.sheep.iterrows():
#   c = int(r.cell)
#   print(r.x, r.y, c, m.grass.loc[c, "x"], m.grass.loc[c,"y"])


#plt.show()
no.run(m)

