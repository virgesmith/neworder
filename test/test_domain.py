import numpy as np
import pandas as pd  # type: ignore
import pytest

import neworder as no


def test_invalid() -> None:
    with pytest.raises(AssertionError):
        no.Space(np.array([]), np.array([]))
    with pytest.raises(AssertionError):
        no.Space(np.array([0.0]), np.array([0.0]))
    with pytest.raises(AssertionError):
        no.Space(np.array([0.0, 1.0]), np.array([1.0, -1.0]))


def test_space2d() -> None:
    # constrained edges
    space2dc = no.Space(np.array([-1.0, -3.0]), np.array([2.0, 5.0]), no.Edge.CONSTRAIN)

    point = np.zeros(2)
    delta = np.array([0.6, 0.7])

    # move point until stuck in corner
    for _ in range(100):
        point, delta = space2dc.move(point, delta, 1.0)

    # check its in corner and not moving
    assert point[0] == 2.0
    assert point[1] == 5.0
    assert delta[0] == 0.0
    assert delta[1] == 0.0

    # wrapped edges
    space2dw = no.Space(np.array([-1.0, -3.0]), np.array([2.0, 5.0]), no.Edge.WRAP)

    assert space2dw.dim == 2

    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    delta = np.array([0.6, 0.7])

    # move point
    for _ in range(100):
        points, delta = space2dw.move(points, delta, 1.0)
        # check distances dont change
        d2, _ = space2dw.dists2(points)
        assert np.all(d2.diagonal() == 0.0)
        assert np.allclose(d2[0], np.array([0.0, 1.0, 1.0]))
        assert np.allclose(d2[1], np.array([1.0, 0.0, 2.0]))

    # check its still in domain and speed unchanged
    assert np.all(points[:, 0] >= -1.0) and np.all(points[:, 0] < 2.0)
    assert np.all(points[:, 1] >= -3.0) and np.all(points[:, 1] < 5.0)
    assert delta[0] == 0.6
    assert delta[1] == 0.7

    # bounce edges
    space2db = no.Space(np.array([-1.0, -3.0]), np.array([2.0, 5.0]), no.Edge.BOUNCE)

    assert space2db.dim == 2

    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    deltas = np.array([[0.6, 0.7], [0.6, 0.7], [0.6, 0.7]])

    # move points
    for _ in range(100):
        points, deltas = space2dw.move(points, deltas, 1.0)

    # check points still in domain and absolute speed unchanged
    assert np.all(points[:, 0] >= -1.0) and np.all(points[:, 0] < 2.0)
    assert np.all(points[:, 1] >= -3.0) and np.all(points[:, 1] < 5.0)
    assert np.all(np.abs(deltas[:, 0]) == 0.6)
    assert np.all(np.abs(deltas[:, 1]) == 0.7)


def test_space3d() -> None:
    rng = np.random.default_rng(19937)

    N = 5
    bodies = pd.DataFrame(
        index=no.df.unique_index(N),
        data={
            "x": rng.random(N) - 0.5,
            "y": rng.random(N) - 0.5,
            "z": rng.random(N) - 0.5,
            "vx": 0.01,
            "vy": 0.01,
            "vz": 0.01,
        },
    )

    space = no.Space.unbounded(3)

    s = np.column_stack((bodies.x, bodies.y, bodies.z))
    assert np.all(space.dists(s).diagonal() == 0.0)

    assert space.dim == 3

    dt = 1.0
    (bodies.x, bodies.y, bodies.z), (bodies.vx, bodies.vy, bodies.vz) = space.move(
        (bodies.x, bodies.y, bodies.z),
        (bodies.vx, bodies.vy, bodies.vz),
        dt,
        ungroup=True,
    )


def test_grid() -> None:
    with pytest.raises(ValueError):
        no.StateGrid(np.empty(shape=(3, 3)), no.Edge.UNBOUNDED)
    with pytest.raises(ValueError):
        no.StateGrid(np.empty(shape=()))
    with pytest.raises(ValueError):
        no.StateGrid(np.empty(shape=(2, 0)))

    state = np.zeros((5, 5))
    state[0, 0] = 1
    state[1, 1] = 2
    state[1, -1] = 3

    # total neighbours should be 3 in corner, 5 on edge, 8 in middle
    g = no.StateGrid(state, no.Edge.CONSTRAIN)
    assert np.sum(g.count_neighbours()) == 3
    assert np.sum(g.count_neighbours(lambda x: x == 2)) == 8
    assert np.sum(g.count_neighbours(lambda x: x == 3)) == 5
    assert np.sum(g.count_neighbours(lambda x: x != 0)) == 16
    assert g.shift((0, 0), (-1, -1)) == (0, 0)

    state = np.zeros((4, 4, 4))
    state[0, 0, 0] = 1
    state[-1, 1, -1] = -1

    # total neighbours should be 26
    g = no.StateGrid(state, no.Edge.WRAP)
    assert np.sum(g.count_neighbours()) == 26
    assert np.sum(g.count_neighbours(lambda x: x == -1)) == 26
    assert np.sum(g.count_neighbours(lambda x: x != 0)) == 52
    assert g.shift((0, 0, 0), (-1, -1, -1)) == (3, 3, 3)

    g = no.StateGrid(state, no.Edge.BOUNCE)
    assert g.shift((0, 0, 0), (-1, -1, -1)) == (1, 1, 1)
