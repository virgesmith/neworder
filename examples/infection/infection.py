import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import matplotlib.animation as animation
import numpy as np
from shapely.geometry import MultiLineString
from shapely.ops import linemerge
from shapely import line_interpolate_point
import geopandas as gpd
import contextily as ctx

NAGENTS = 10
SPEED = 20

plt.ion()
np.random.seed(19937)

G = ox.graph_from_point((53.925, -1.822), dist=2000, network_type="drive")
G = ox.project_graph(G, to_crs='epsg:27700')
all_nodes, all_edges = ox.graph_to_gdfs(G)

def traverse(graph, start, ends):
    # find the best path from start to end
    paths = []
    print(ends)
    for end in ends:
        nodes = nx.shortest_path(graph, source=start, target=end, weight='length')
        edges = zip(nodes[:-1], nodes[1:], [0] * (len(nodes) - 1))

        segments = all_edges.loc[edges, "geometry"].values
        paths.append(linemerge(MultiLineString(segments)))

    return paths


def random_next(node: int) -> int:
    candidates = [edge for edge in G.out_edges(node)] # if edge not in path and (edge[1], edge[0]) not in path]
    if candidates:
        return candidates[np.random.choice(len(candidates))][1]
    else:
        return node

start_positions = all_nodes.sample(n=NAGENTS).index.values
agents = gpd.GeoDataFrame(data={"node": start_positions, "speed": SPEED, "infected": [True] + [False] * (len(start_positions) - 1)})
agents["dest"] = agents["node"].apply(random_next)
agents["dist"] = agents[["node", "dest"]].apply(lambda r: all_edges.loc[(r["node"], r["dest"], 0), "length"], axis=1)
agents["offset"] = 0
agents["path"] = agents[["node", "dest"]].apply(lambda r: all_edges.loc[(r["node"], r["dest"], 0), "geometry"], axis=1)
agents["geometry"] = agents["path"].apply(lambda linestr: line_interpolate_point(linestr, 0))

def update(agents: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    agents.offset += agents.speed
    agents["geometry"] = agents[["path", "offset", "speed"]].apply(lambda r: line_interpolate_point(r["path"], r["offset"]), axis=1)
    overshoots = agents.offset >= agents.dist
    if not overshoots.empty:
        # offset <- offset - dist
        agents.loc[overshoots, "offset"] -= agents.loc[overshoots, "dist"]
        # node <- dest
        agents.loc[overshoots, "node"] = agents.loc[overshoots, "dest"]
        # dest <- random
        agents.loc[overshoots, "dest"] = agents.loc[overshoots, "node"].apply(random_next)
        # path <- (node, dest), dist <- new_dist
        agents.loc[overshoots, "path"] = agents[["node", "dest"]].apply(lambda r: all_edges.loc[(r["node"], r["dest"], 0), "geometry"], axis=1)
        agents.loc[overshoots, "dist"] = agents[["node", "dest"]].apply(lambda r: all_edges.loc[(r["node"], r["dest"], 0), "length"], axis=1)
        # position <- update(path, offset)
        agents.loc[overshoots, "geometry"] = agents.loc[overshoots, ["path", "offset", "speed"]].apply(lambda r: line_interpolate_point(r["path"], r["offset"]), axis=1)
    return agents

def infect(agents):
    infected = agents[agents.infected].geometry.values
    for i in infected:
        agents.loc[agents.geometry.distance(i) < SPEED, "infected"] = True
    return agents

_fig, ax = ox.plot_graph(G, bgcolor="w", node_size=5, edge_linewidth=2, edge_color="#777777", figsize=(12,9))
plt.tight_layout()
ctx.add_basemap(ax, crs=all_nodes.crs, url=ctx.providers.OpenTopoMap)

def update_position(n: int):
    global agents
    agents = infect(update(agents))
    offsets = np.array(list(zip(agents.geometry.x, agents.geometry.y)))
    colours = agents.infected.apply(lambda c: [1.0, 0.0, 0.0, 1.0] if c else [0.0, 0.0, 0.0, 1.0])
    axa.set_offsets(offsets)
    axa.set_facecolors(colours)

axa = ax.scatter(agents.geometry.x, agents.geometry.y, color=agents.infected.apply(lambda x: 'r' if x else 'k'))
ani = animation.FuncAnimation(_fig, update_position, interval=0, repeat=False)

plt.ioff()
plt.show()

