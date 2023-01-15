from typing import Any
from enum import Enum
from time import sleep
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
from shapely import line_interpolate_point
import geopandas as gpd
# import contextily as ctx

import neworder as no

class Status(Enum):
  SUSCEPTIBLE = 0
  INFECTED = 1
  IMMUNE = 2
  DEAD = 3

  @property
  def rgba(self):
    match self:
      case Status.SUSCEPTIBLE:
        return [1.0, 1.0, 1.0, 1.0]
      case Status.INFECTED:
        return [1.0, 0.0, 0.0, 1.0]
      case Status.IMMUNE:
        return [0.0, 1.0, 0.0, 1.0]
      case Status.DEAD:
        return [0.0, 0.0, 0.0, 1.0]


class Infection(no.Model):
  def __init__(self, n_agents: int, n_infected: int, speed: float, infection_radius: float, recovery_time: int, mortality: float) -> None:
    super().__init__(no.LinearTimeline(0.0, 1.0), no.MonteCarlo.deterministic_independent_stream)
    self.nprand = no.as_np(self.mc)
    G = ox.graph_from_point((53.925, -1.822), dist=2000, network_type="drive")
    self.G = ox.project_graph(G, to_crs='epsg:27700')
    self.nodes, self.edges = ox.graph_to_gdfs(self.G)
    self.infection_radius = infection_radius
    self.recovery_time = recovery_time
    self.marginal_mortality = 1 - (1 - mortality) ** (1/recovery_time)

    start_positions = self.nodes.sample(n=n_agents, random_state=self.nprand, replace=True).index.values
    agents = gpd.GeoDataFrame(data={"node": start_positions, "speed": speed, "status": Status.SUSCEPTIBLE, "t_infect": no.time.never()})
    agents["dest"] = agents["node"].apply(self.__random_next_dest)
    agents["dist"] = agents[["node", "dest"]].apply(lambda r: self.edges.loc[(r["node"], r["dest"], 0), "length"], axis=1)
    agents["offset"] = 0
    agents["path"] = agents[["node", "dest"]].apply(lambda r: self.edges.loc[(r["node"], r["dest"], 0), "geometry"], axis=1)
    agents["geometry"] = agents["path"].apply(lambda linestr: line_interpolate_point(linestr, 0))
    infected = self.nprand.choice(agents.index, n_infected, replace=False)
    agents.loc[infected, "status"] = Status.INFECTED
    agents.loc[infected, "t_infect"] = self.timeline.index()

    self.agents = agents
    self.fig, self.g = self.__init_visualisation()

  def step(self) -> None:
    self.__update_position()
    self.__infect_nearby()
    self.__recover()
    self.__succumb()
    num_infected = (self.agents.status == Status.INFECTED).sum()
    num_immune = (self.agents.status == Status.IMMUNE).sum()
    num_dead = (self.agents.status == Status.DEAD).sum()
    i = self.timeline.index()
    if i % 10 == 0:
        no.log(f"step {i}: inf:{num_infected} imm:{num_immune} dead:{num_dead} / {len(self.agents)}")
    self.__update_visualisation()
    if num_infected == 0:
        sleep(5)
        self.halt()
        self.finalise()

  def finalise(self) -> None:
    no.log(f"total steps: {self.timeline.index()}")
    no.log(f"infections: {len(self.agents.t_infect.dropna())}")
    no.log(f"recoveries: {(self.agents.status == Status.IMMUNE).sum()}")
    no.log(f"deaths: {(self.agents.status == Status.DEAD).sum()}")
    no.log(f"unaffected: {(self.agents.status == Status.SUSCEPTIBLE).sum()}")

  def __random_next_dest(self, node: int) -> int:
    candidates = [edge for edge in self.G.out_edges(node)] # if edge not in path and (edge[1], edge[0]) not in path]
    if candidates:
        return candidates[self.nprand.choice(len(candidates))][1]
    else:
        return node

  def __update_position(self) -> None:
    self.agents.offset += self.agents.speed
    self.agents["geometry"] = self.agents[["path", "offset", "speed"]].apply(lambda r: line_interpolate_point(r["path"], r["offset"]), axis=1)
    overshoots = self.agents.offset >= self.agents.dist
    if not overshoots.empty:
      # offset <- offset - dist
      self.agents.loc[overshoots, "offset"] -= self.agents.loc[overshoots, "dist"]
      # node <- dest
      self.agents.loc[overshoots, "node"] = self.agents.loc[overshoots, "dest"]
      # dest <- random
      self.agents.loc[overshoots, "dest"] = self.agents.loc[overshoots, "node"].apply(self.__random_next_dest)
      # path <- (node, dest), dist <- new_dist
      self.agents.loc[overshoots, "path"] = self.agents[["node", "dest"]].apply(lambda r: self.edges.loc[(r["node"], r["dest"], 0), "geometry"], axis=1)
      self.agents.loc[overshoots, "dist"] = self.agents[["node", "dest"]].apply(lambda r: self.edges.loc[(r["node"], r["dest"], 0), "length"], axis=1)
      # position <- update(path, offset)
      self.agents.loc[overshoots, "geometry"] = self.agents.loc[overshoots, ["path", "offset", "speed"]].apply(lambda r: line_interpolate_point(r["path"], r["offset"]), axis=1)

  def __infect_nearby(self):
    infected = self.agents[self.agents.status == Status.INFECTED].geometry
    susceptible = self.agents[self.agents.status == Status.SUSCEPTIBLE].geometry
    new_infections = []
    # loop over smallest group
    if len(infected) < len(susceptible):
      for i in infected:
        new = susceptible.geometry.distance(i) < self.infection_radius
        # new[new].index gives us only the index values corresponding to True
        new_infections.extend(new[new].index)
    else:
      for i, p in susceptible.items():
        new = infected.geometry.distance(p) < self.infection_radius
        if new.any():
          new_infections.append(i)
    self.agents.loc[new_infections, "status"] = Status.INFECTED
    self.agents.loc[new_infections, "t_infect"] = self.timeline.index()

  def __recover(self) -> None:
    t = self.timeline.index()
    self.agents.loc[(t - self.agents.t_infect >= self.recovery_time) & (self.agents.status == Status.INFECTED), "status"] = Status.IMMUNE

  def __succumb(self) -> None:
    infected = self.agents[self.agents.status == Status.INFECTED]
    death = self.mc.hazard(self.marginal_mortality, len(infected)).astype(bool)
    # if death.sum():
    #   print(infected[death].index.values)
    #   print(self.agents.loc[623])
    self.agents.loc[infected[death].index.values, "status"] = Status.DEAD
    self.agents.loc[infected[death].index.values, "speed"] = 0.0


  def __init_visualisation(self) -> tuple[Any, Any]:
    plt.ion()
    fig, ax = ox.plot_graph(self.G, bgcolor="w", node_size=5, edge_linewidth=2, edge_color="#777777", figsize=(12,9))
    plt.tight_layout()
    # ctx.add_basemap(ax, crs=self.nodes.crs, url=ctx.providers.OpenTopoMap)
    g = ax.scatter(self.agents.geometry.x, self.agents.geometry.y, color=self.agents.status.apply(lambda c: c.rgba), edgecolor='k')
    fig.suptitle("[q to quit]")
    fig.canvas.mpl_connect('key_press_event', lambda event: self.halt() if event.key == "q" else None)
    fig.canvas.flush_events()
    return fig, g

  def __update_visualisation(self) -> None:
    offsets = np.array(list(zip(self.agents.geometry.x, self.agents.geometry.y)))
    colours = self.agents.status.apply(lambda c: c.rgba)
    self.g.set_offsets(offsets)
    self.g.set_facecolors(colours)
    self.fig.canvas.flush_events()

