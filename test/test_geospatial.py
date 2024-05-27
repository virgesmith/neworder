import pytest


def test_geospatial() -> None:
    geospatial = pytest.importorskip("neworder.geospatial")
    # TODO...
    domain = geospatial.GeospatialGraph.from_point(
        (54.3748, -2.9988), dist=2000, network_type="drive", crs="epsg:27700"
    )

    assert domain.crs == "epsg:27700"
    assert len(domain.graph) == len(domain.all_nodes)
