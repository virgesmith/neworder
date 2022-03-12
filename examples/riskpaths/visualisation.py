from matplotlib import pyplot as plt  # type: ignore
import neworder
from data import min_age, max_age

def plot(model: neworder.Model) -> None:

  bins=range(int(min_age),int(max_age)+1)

  b = [ model.population.T_Union1Start,
        model.population.T_Union1End,
        model.population.T_Union2Start,
        model.population.T_Union2End ]

  plt.hist(b, bins=bins, stacked=True)
  plt.hist(model.population.TimeOfPregnancy, bins, color='purple')
  plt.legend(["Union 1 starts", "Union 1 ends", "Union 2 starts", "Union 2 ends", "Pregnancy"])
  plt.title("Age distribution of first pregnancy, dependent on union state")
  plt.xlabel("Age (y)")
  plt.ylabel("Frequency")
  #plt.savefig("./docs/examples/img/riskpaths.png", dpi=80)
  plt.show()
