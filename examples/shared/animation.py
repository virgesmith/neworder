import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Hist:
  def __init__(self, data, numbins, filename=None):

    fig, ax = plt.subplots()
    # TODO histtype='barstacked'
    # see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html
    self.n, bins, self.patches = plt.hist(data, numbins, facecolor='black')

    ax.set_title("Case-based mortality microsim (%d people)" % len(data))
    ax.set_xlabel("Age at Death")
    ax.set_ylabel("Persons")

    frames = numbins+1
    anim = animation.FuncAnimation(fig, self.__animate, interval=100, frames=numbins, repeat=True, repeat_delay=3000)

    plt.show()
    if filename is not None:
      anim.save(filename, dpi=80, writer='imagemagick') 

  def __animate(self, frameno):
    i = 0
    for rect, h in zip(self.patches, self.n):
      rect.set_height(h if i <= frameno else 0)
      i = i + 1
    return self.patches

# N, mu, sigma = 10000, 100, 15
# x = mu + sigma * np.random.randn(N) 
# # y,x = np.histogram(x, bins=100)
# Hist(x, 100)#, "hist.gif")

