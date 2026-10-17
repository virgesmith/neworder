import numpy as np
import pandas as pd
import pytest

import neworder as no


def test_invalid() -> None:
    with pytest.raises(ValueError):
        no.Space(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        no.Space(np.array([0.0]), np.array([0.0]))
    with pytest.raises(ValueError):
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
        points, deltas = space2db.move(points, deltas, 1.0)

    # check points still in domain and absolute speed unchanged
    assert np.all(points[:, 0] >= -1.0) and np.all(points[:, 0] <= 2.0)
    assert np.all(points[:, 1] >= -3.0) and np.all(points[:, 1] <= 5.0)
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
    (bodies.x, bodies.y, bodies.z), (bodies.vx, bodies.vy, bodies.vz) = space.move(  # ty:ignore[unresolved-attribute]
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


def test_space_validation() -> None:
    with pytest.raises(ValueError):
        no.Space.unbounded(0)
    with pytest.raises(ValueError):
        no.Space(np.zeros(2), np.ones(2), edge="wrap")  # ty:ignore[invalid-argument-type]

    space = no.Space(np.zeros(2), np.ones(2))
    assert space.continuous
    assert space.edge == no.Edge.CONSTRAIN
    lo, hi = space.extent
    assert np.array_equal(lo, np.zeros(2)) and np.array_equal(hi, np.ones(2))
    assert repr(space) == "Space dim=2 min=[0. 0.] max=[1. 1.] edge=Edge.CONSTRAIN"

    ints = np.zeros((1, 2), dtype=int)
    with pytest.raises(TypeError):
        space.move(ints, np.zeros((1, 2)), 1.0)
    with pytest.raises(TypeError):
        space.move(np.zeros((1, 2)), ints, 1.0)
    with pytest.raises(ValueError):
        space.move(np.zeros((1, 3)), np.zeros((1, 3)), 1.0)
    with pytest.raises(TypeError):
        space.dists2(ints)
    with pytest.raises(TypeError):
        space.dists2(np.zeros((1, 2)), ints)


def test_space_bounce() -> None:
    space = no.Space(np.zeros(2), np.ones(2), no.Edge.BOUNCE)

    # crosses the max edge in x and the min edge in y
    p, v = space.move(np.array([[0.9, 0.1]]), np.array([[0.2, -0.3]]), 1.0)
    assert np.allclose(p, [[0.9, 0.2]])
    assert np.allclose(v, [[-0.2, 0.3]])


def test_space_in_range() -> None:
    space = no.Space(np.zeros(2), np.full(2, 10.0))
    x = np.array([1.0, 2.0, 5.0])
    y = np.ones(3)

    d2, (dx, dy) = space.dists2((x, y), (x[:1], y[:1]))
    assert np.array_equal(d2, [[0.0, 1.0, 16.0]])
    assert np.array_equal(dx, [[0.0, 1.0, 4.0]])
    assert np.array_equal(dy, np.zeros((1, 3)))

    points = np.column_stack((x, y))
    # excludes each point itself
    assert np.array_equal(space.in_range(2.0, points), [[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    assert np.array_equal(space.in_range(2.0, points, count=True), [1, 1, 0])
    # explicit to_points are not excluded
    assert np.array_equal(space.in_range(2.0, points, points[:1]), [[1, 1, 0]])
    assert np.array_equal(space.in_range(2.0, points, points[:1], count=True), [2])


def test_grid_access() -> None:
    g = no.StateGrid(np.zeros((3, 4)))
    assert not g.continuous
    assert g.extent == (3, 4)

    g[1, 2] = 5
    assert g[1, 2] == 5
    assert g.state[1, 2] == 5
    # out-of-range points are clipped to the edge
    g[5, 9] = 7
    assert g.state[2, 3] == 7
    assert g[-1, -1] == g.state[0, 0]

    with pytest.raises(ValueError):
        _ = g[1, 2, 3]
