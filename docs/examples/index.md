# Examples

Runnable microsimulation and agent-based models, ranging from a minimal single-step model to full parallel population dynamics. Each card links to a walkthrough of that example's implementation; the source for all of them lives under [`examples/`](https://github.com/virgesmith/neworder/tree/main/examples) in the repository.

{{ include_snippet("./docs/examples/src.md", show_filename=False) }}

<div class="grid cards" markdown>

-   :material-hand-wave:{ .lg .middle } __Hello World__

    ---

    The minimal model: framework structure, workflow and execution, start to finish.

    [:octicons-arrow-right-24: Walkthrough](hello-world.md)

-   :material-server-network:{ .lg .middle } __Parallel Execution__

    ---

    Exchanging and synchronising data between MPI processes (and multithreaded execution).

    [:octicons-arrow-right-24: Walkthrough](parallel.md)

-   :material-transit-connection-variant:{ .lg .middle } __Markov Chain__

    ---

    Probabilistic transitions between discrete states, and fast dataframe-based sampling.

    [:octicons-arrow-right-24: Walkthrough](markov-chain.md)

-   :material-book-open-page-variant:{ .lg .middle } __Chapter 1__

    ---

    A basic cohort model of mortality with a constant hazard rate, from *Microsimulation and Population Dynamics*.

    [:octicons-arrow-right-24: Walkthrough](chapter1.md)

-   :material-chart-bell-curve:{ .lg .middle } __Mortality__

    ---

    The Life Table example, implemented two ways - discrete case-based vs. continuous sampling - to compare performance.

    [:octicons-arrow-right-24: Walkthrough](mortality.md)

-   :material-dice-multiple:{ .lg .middle } __Membership__

    ---

    An open population with members joining and churning, showcasing a hash-based random stream stable under sub-sampling and reordering.

    [:octicons-arrow-right-24: Walkthrough](membership.md)

-   :material-scale-balance:{ .lg .middle } __Competing Risks__

    ---

    Case-based simulation of competing fertility and mortality events as nonhomogeneous Poisson processes.

    [:octicons-arrow-right-24: Walkthrough](competing.md)

-   :material-family-tree:{ .lg .middle } __RiskPaths__

    ---

    A well-known MODGEN teaching model of fertility as a function of time and union state.

    [:octicons-arrow-right-24: Walkthrough](riskpaths.md)

-   :material-account-group:{ .lg .middle } __Population Microsimulation__

    ---

    A full demographic model: fertility, mortality and migration acting on a real population pyramid.

    [:octicons-arrow-right-24: Walkthrough](people.md)

-   :material-finance:{ .lg .middle } __Derivative Pricing__

    ---

    Parallel runs with identical random streams but perturbed inputs, to compute pricing sensitivities.

    [:octicons-arrow-right-24: Walkthrough](option.md)

-   :material-grid-large:{ .lg .middle } __Conway's Game of Life__

    ---

    The classic cellular automaton.

    [:octicons-arrow-right-24: Walkthrough](conway.md)

-   :material-map-marker-distance:{ .lg .middle } __Schelling's Segregation Model__

    ---

    A classic agent-based model of residential segregation from simple neighbour preferences.

    [:octicons-arrow-right-24: Walkthrough](schelling.md)

-   :material-paw:{ .lg .middle } __Wolf-Sheep Predation__

    ---

    Another classic agent-based model, of predator-prey population dynamics.

    [:octicons-arrow-right-24: Walkthrough](wolf-sheep.md)

-   :material-bird:{ .lg .middle } __Boids__

    ---

    Collective flocking behaviour emerging from simple local interaction rules.

    [:octicons-arrow-right-24: Walkthrough](boids.md)

-   :material-virus:{ .lg .middle } __Infection Model__

    ---

    Individuals moving and interacting - and transmitting infection - on a geospatial network.

    [:octicons-arrow-right-24: Walkthrough](infection.md)

</div>
