import neworder as no
import osmnx as ox
#import pytest

import matplotlib.pyplot as plt

def test_geospatial() -> None:
    # TODO...
    domain = no.GeospatialGraph.from_point((54.3748, -2.9988), dist=1000, network_type="drive", crs='epsg:27700')
    assert domain.crs == "epsg:27700"
    assert len(domain.graph)
    assert len(domain.all_edges)
    assert len(domain.all_nodes)


def isochrone() -> None:
    domain = no.GeospatialGraph.from_point((54.3748,  -2.9988), dist=2000, network_type="drive", crs='epsg:27700')
    ax = domain.all_edges.plot(figsize=(10,10), color="grey")

    plt.tight_layout()
    origin = domain.all_nodes.index[10]
    # domain.all_nodes.loc[[origin]].plot(ax = ax, color="r")
    # ox.plot_graph(
    subgraph = domain.subgraph(origin, radius=500.0, distance="length")
    subnodes, _ = ox.graph_to_gdfs(subgraph)
    ax.scatter(subnodes.geometry.x, subnodes.geometry.y, color="k")

    polygon = domain.isochrone(origin, radius=500.0, distance="length")
    ax.plot(*polygon.exterior.xy, color="g")

    plt.show()


if __name__ == "__main__":
    isochrone()



