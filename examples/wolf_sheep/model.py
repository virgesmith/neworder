from wolf_sheep import WolfSheep

import neworder as no

# no.verbose()

assert (
    no.mpi.RANK == 0 and no.mpi.SIZE == 1
), "this example should only be run in serial mode"

params = {
    "grid": {"width": 100, "height": 100},
    "wolves": {
        "starting_population": 100,
        "reproduce": 0.05,
        "speed": 2.5,
        "speed_variance": 0.05,
        "gain_from_food": 20,
    },
    "sheep": {
        "starting_population": 300,
        "reproduce": 0.04,
        "speed": 0.9,
        "speed_variance": 0.02,
        "gain_from_food": 4,
    },
    "grass": {"regrowth_time": 12},
}

m = WolfSheep(params)

no.run(m)
