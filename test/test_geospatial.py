import os
from pathlib import Path

import pytest

geospatial = pytest.importorskip("neworder.geospatial")

import networkx as nx  # noqa: E402
import osmnx as ox  # noqa: E402
import requests  # noqa: E402
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError  # noqa: E402
from shapely.geometry import LineString, Polygon  # noqa: E402

# Drive network within 2km of (54.3748, -2.9988), saved with ox.save_graphml so tests don't depend on the Overpass API
GRAPH_FILE = Path(__file__).parent / "coniston.graphml"
ORIGIN = (54.3748, -2.9988)


@pytest.fixture(scope="module")
def domain() -> "geospatial.GeospatialGraph":
    return geospatial.GeospatialGraph(ox.load_graphml(GRAPH_FILE), crs="epsg:27700")


def test_geospatial(domain: "geospatial.GeospatialGraph") -> None:
    assert domain.crs == "epsg:27700"
    assert len(domain.graph) == len(domain.all_nodes) == 69
    assert domain.graph.number_of_edges() == len(domain.all_edges) == 161


def test_geospatial_edges(domain: "geospatial.GeospatialGraph") -> None:
    for node in domain.graph.nodes:
        assert all(u == node for u, _ in domain.edges_from(node))
        assert all(v == node for _, v in domain.edges_to(node))


def test_geospatial_routing(domain: "geospatial.GeospatialGraph") -> None:
    origin = next(iter(domain.graph.nodes))
    dest = max(nx.single_source_dijkstra_path_length(domain.graph, origin, weight="length").items(), key=lambda x: x[1])

    path = domain.shortest_path(origin, dest[0], weight="length")
    assert isinstance(path, LineString)
    assert path.length == pytest.approx(dest[1], rel=0.01)

    subgraph = domain.subgraph(origin, radius=500, distance="length")
    assert origin in subgraph
    assert 1 < len(subgraph) < len(domain.graph)

    isochrone = domain.isochrone(origin, radius=500, distance="length")
    assert isinstance(isochrone, Polygon)
    assert isochrone.area > 0


# Queries the live Overpass API, which rate-limits and is sometimes unavailable, so this only runs when explicitly
# enabled (on a single CI job) and skips rather than fails on network errors
@pytest.mark.skipif(not os.getenv("NEWORDER_NETWORK_TESTS"), reason="set NEWORDER_NETWORK_TESTS=1 to enable")
def test_geospatial_from_point() -> None:
    try:
        domain = geospatial.GeospatialGraph.from_point(ORIGIN, dist=2000, network_type="drive", crs="epsg:27700")
    except (requests.RequestException, ResponseStatusCodeError, InsufficientResponseError) as e:
        pytest.skip(f"Overpass API unavailable: {e}")

    assert domain.crs == "epsg:27700"
    assert len(domain.graph) == len(domain.all_nodes) > 0
