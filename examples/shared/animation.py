import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Hist:
  def __init__(self, data, numbins):

    fig, ax = plt.subplots()
    # see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html
    self.n, _bins, self.patches = plt.hist(data, numbins, facecolor='black')

    ax.set_title("Case-based mortality microsim (%d people)" % len(data))
    ax.set_xlabel("Age at Death")
    ax.set_ylabel("Persons")

    self.anim = animation.FuncAnimation(fig, self.__animate, interval=100, frames=numbins, repeat=True, repeat_delay=3000)

  def save(self, filename):
    self.anim.save(filename, dpi=80, writer='imagemagick')

  def show(self):
    plt.show()

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

