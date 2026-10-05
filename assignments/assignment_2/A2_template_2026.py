"""EC A2 template code - neuroevolution for targeted locomotion with ARIEL.

WHAT THIS FILE IS
-----------------
A *demo*, not a solution. It spawns a robot, drives it with a neural network
whose weights are RANDOM, runs the simulation, and reports how close the robot
ended up to a target.

There is deliberately NO evolution in here. Building the EA (representation,
initialisation, parent selection, variation, survivor selection) is the assignment.
See "YOUR JOB" at the bottom of this file.

THE ASSIGNMENT IN A NUTSHELL
------------------------------
Evolve the weights of a neural network controller so that a robot moves from
SPAWN_POS to TARGET_POSITION within the simulation time.

    fitness = distance between the robot's final position and TARGET_POSITION

HOW TO RUN
----------

Change MODE below to switch between an interactive viewer, a headless run,
a rendered video, or a single frame.
"""
import argparse
import csv
from datetime import datetime
# Standard library
from pathlib import Path
from typing import Literal, cast

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
from networkx.classes import number_of_nodes

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import centipede_3
from ariel.ec import set_seed, config, Population, Individual, EAOperation, EA, Crossover
from ariel.simulation.environments import SimpleFlatWorld, OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

# Type aliases
type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
SEED = 42
RNG = np.random.default_rng(SEED)

# ariel.ec's own generators/mutators/crossover draw from a separate,
# package-level RNG. Reseed it too if you build your EA on ariel.ec,
# or every one of your "multiple seeds" runs the same variation operators.
set_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M_%S_%f")

DATA = CWD / "__data__" / SCRIPT_NAME / f"seed_{SEED}_{RUN_ID}"
FITNESS_CSV = DATA / "fitness_data" / f"fitness.csv"

# --- EXPERIMENT CONSTANTS --- #
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]  # where the robot starts
#TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]  # where it should end up
TARGET_POSITION: list[float] = [5.5, 0.0, 0.1]
SIM_DURATION: float = 15.0  # seconds of simulated time per evaluation
MODE: ViewerTypes = "launcher"  # see run_experiment() for the options


# ============================================================================ #
#  1. THE BODY AND THE WORLD
# ============================================================================ #
def build_world() -> OlympicArena:
    """Create the environment the robot lives in.

    YOU MAY CHANGE THIS. Options include: SimpleFlatWorld, RuggedTerrainWorld,
    CraterTerrainWorld, AmphitheatreTerrainWorld, OlympicArena, ...
    (SimpleTiltedWorld is not supported for this task.)

    Whatever you pick, keep it FIXED for all runs you compare against each
    other, and say in your report which one you used. A controller evolved on
    flat ground and one evolved on rugged terrain are not comparable numbers.
    """
    world = OlympicArena()
    world.spec.worldbody.add_site(
        name="target_marker",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        pos=[
            TARGET_POSITION[0],
            TARGET_POSITION[1],
            1.01
        ],
        size=[TARGET_RADIUS, 0.01, 0.0],
        rgba=[0.1, 0.9, 0.1, 0.7]
    )
    return world
    #return OlympicArena()


def build_robot() -> CoreModule:
    """Create the robot body.

    YOU MAY CHANGE THIS. Options include the prebuilt bodies in
    `ariel.body_phenotypes.robogen_lite.prebuilt_robots` (gecko, spider, ...).

    Two consequences of this choice, and they matter:
      * The body determines `model.nu` (the number of hinges you must send
        commands to) - that is the OUTPUT size of your controller.
      * The body determines the size of `data.qpos` - if you feed qpos to your
        network, that is (part of) your INPUT size.
    Change the body and your genotype length changes with it. Keep the body
    FIXED within an experiment.
    """
    return centipede_3()


# ============================================================================ #
#  2. THE CONTROLLER CONTRACT
# ============================================================================ #
#
# MuJoCo calls the controller every physics step with (model, data); its job
# is to write into data.ctrl.
#
#   INPUTS   : whatever you read from `data` (qpos, qvel, time, ...), plus any
#              task info you already know, e.g. the vector to TARGET_POSITION.
#              INPUT SIZE is your choice, but must stay CONSTANT.
#   OUTPUTS  : exactly `model.nu` values, one per actuated hinge.
#   RANGE    : hinges accept [-pi/2, +pi/2] radians. A tanh output gives
#              [-1, 1] - rescale: actions * (np.pi / 2).
#   WRITING  : DIRECT (data.ctrl[:] = actions) commands the angle straight -
#              fast, but can destabilise the sim on large jumps. DELTA
#              (data.ctrl[:] += actions * alpha, alpha ~ 0.05, then clip) is
#              smoother but accumulates, so clipping is required. Pick one,
#              justify it, use it everywhere.
#   NaN      : blown-up weights silently write NaN into data.ctrl. Assert
#              against it while developing.
#
# ============================================================================ #

# Controller architecture - decide before writing your EA.
HIDDEN_SIZE: int = 6


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    weights: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Map robot state to hinge commands: in -> hidden -> actions.

    In this demo `weights` is drawn at RANDOM. In your assignment, `weights`
    is what the evolutionary algorithm produces: an individual's genotype,
    reshaped into these matrices. You are free to change the architecture
    itself (layers, activations, ...) - just keep input/output sizes correct.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model. Use `model.nu` for the number of hinges.
    data : mj.MjData
        The MuJoCo data. This is where you read the robot's state from.
    weights : list of ndarray
        [w1, w2] - the layer weight matrices.

    Returns
    -------
    npt.NDArray[np.float64]
        `model.nu` action values, already scaled to [-pi/2, pi/2].
    """
    w1, w2 = weights

    # --- INPUTS ---------------------------------------------------------- #
    # Bare qpos - the simplest choice, not necessarily a good one. See
    # YOUR JOB below.
    inputs = data.qpos

    # --- FORWARD PASS ----------------------------------------------------- #
    layer1 = np.tanh(inputs @ w1)
    outputs = np.tanh(layer1 @ w2)  # in [-1, 1]

    # --- RESCALE TO THE HINGE RANGE --------------------------------------- #
    return outputs * (np.pi / 2)  # in [-pi/2, pi/2]


def make_random_weights(
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Draw a random parameter set for `nn_controller`.

    THIS IS THE FUNCTION YOUR EA REPLACES. Instead of sampling weights from a
    normal distribution, your EA will search for them.

    Note the total parameter count printed by main(): that is the length of the
    flat vector an individual's genotype has to encode. Reshaping a flat
    genotype back into these matrices is on you.
    """
    return [
        RNG.normal(scale=0.5, size=(input_size, HIDDEN_SIZE)),
        RNG.normal(scale=0.5, size=(HIDDEN_SIZE, output_size)),
    ]


# ============================================================================ #
#  3. POSITION AND FITNESS
# ============================================================================ #
#
# The robot is spawned with a free joint, so data.qpos[0:3] IS the core's
# (x, y, z) world position. Read it before and after stepping - no tracker or
# bookkeeping needed. (`data.geom("robot1_core").xpos` works too.)
#
# ============================================================================ #

TARGET_RADIUS = 0.15

STILL_RADIUS = 0.1
STILL_PENALTY = 0.5
INVALID_FITNESS = 1_000_000.0

FINAL_DISTANCE_WEIGHT = 0.75
MEAN_DISTANCE_WEIGHT = 0.25

def _distance(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    return float(np.linalg.norm(x[:2] - y[:2]))

def _fitness_when_target_reached(time_to_target: float | None, duration: float) -> float:
    if time_to_target is None:
        return 1.0
    return float(
        np.clip(
                time_to_target / max(duration, 1e-6),
                0.0, 1.0
        )
    )

def _fitness_when_target_not_reached(
    initial_distance: float,
    final_distance: float,
    mean_distance: float
) -> float:
    distance_scale = max(initial_distance, 1e-6)

    normalized_final = final_distance / distance_scale
    normalized_mean = mean_distance / distance_scale

    return float(
        1.0
        + FINAL_DISTANCE_WEIGHT * normalized_final
        + MEAN_DISTANCE_WEIGHT * normalized_mean
    )

def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the robot core's current (x, y, z) world position."""
    return np.asarray(data.qpos[0:3]).copy()


def fitness_function(
    initial_position: npt.NDArray[np.float64],
    final_position: npt.NDArray[np.float64],
    target_position: npt.NDArray[np.float64],
    mean_distance: float,
    time_to_target: float | None,
    duration: float,
) -> float:
    """Score one evaluation. LOWER IS BETTER.

    The plain version: how far is the robot from the target when time runs out?

    `initial_position` is unused here on purpose - it is passed in because the
    moment you want a less naive fitness you will need it. Some things worth
    thinking about (and, ideally, comparing in your report):
      * Distance *reduced* rather than distance remaining, so a robot that
        starts closer is not rewarded for standing still.
      * Penalising a robot that falls over or leaves the arena.
      * Whether the z-axis should count at all - a robot that jumps is not
        closer to the target in any way you care about.
    See `ariel.simulation.tasks.targeted_locomotion` for some worked variants.
    """
    #target = np.asarray(TARGET_POSITION)
    #return float(np.linalg.norm(final_position[:2] - target[:2]))
    initial_distance = _distance(initial_position, target_position)
    final_distance = _distance(final_position, target_position)

    values = [
        initial_distance,
        final_distance,
        mean_distance,
        duration
    ]

    if not np.all(np.isfinite(values)):
        return INVALID_FITNESS
    if duration <= 0.0:
        return INVALID_FITNESS

    reached_target = final_distance <= TARGET_RADIUS
    if reached_target:
        fitness = _fitness_when_target_reached(time_to_target, duration)
    else:
        fitness = _fitness_when_target_not_reached(initial_distance, final_distance, mean_distance)

    return float(fitness)

# ============================================================================ #
#  4. RUNNING ONE EVALUATION
# ============================================================================ #


def run_experiment(genome: list[float], mode: ViewerTypes = MODE) -> float:
    """Set up the world, run one simulation, and return the fitness.

    This is the function your EA calls once per individual, with `mode` set
    to "simple" (headless).

    Returns
    -------
    float
        The fitness of this run. Lower is better.
    """
    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    # --- World and robot --------------------------------------------------- #
    world = build_world()
    robot = build_robot()

    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Compile the world into a model. USE AS IS.
    model = world.spec.compile()
    data = mj.MjData(model)

    # Put the simulation in a clean, known state before reading anything.
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # --- Wire up the controller -------------------------------------------- #
    # Sizes are read from the compiled model, never hardcoded - they depend on
    # the body you chose in build_robot().
    input_size = len(data.qpos)
    output_size = model.nu

    weights = decode_genome(genome, input_size, output_size)
    target = np.asarray(TARGET_POSITION, dtype=np.float64)

    initial_position = get_core_position(data)
    max_displacement = 0.0
    initial_distance = _distance(initial_position, target)
    start_time = float(data.time)

    last_time = start_time
    last_distance = initial_distance

    distance_integral = 0.0
    time_to_target: float | None = None
    if initial_distance <= TARGET_RADIUS:
        time_to_target = 0.0

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        """Compute and apply actions; MuJoCo calls this every physics step."""
        nonlocal last_time
        nonlocal last_distance
        nonlocal distance_integral
        nonlocal time_to_target
        nonlocal max_displacement

        actions = nn_controller(m, d, weights)

        # DIRECT application (see the controller contract above).
        # d.ctrl[:] = actions

        # DELTA application - comment out the line above and use these instead:
        delta = 0.05
        d.ctrl[:] += actions * delta
        d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    # --- Record the starting point ----------------------------------------- #
        current_time = float(d.time)
        dt = current_time - last_time
        if dt <= 0.0:
            return

        current_position = get_core_position(d)
        max_displacement = max(max_displacement, _distance(current_position, initial_position))
        current_distance = _distance(current_position, target)
        distance_integral += (
            0.5
            * (last_distance + current_distance)
            * dt
        )

        if time_to_target is None and current_distance <= TARGET_RADIUS:
            time_to_target = current_time - start_time

        last_time = current_time
        last_distance = current_distance

    # --- Run ---------------------------------------------------------------- #
    if mode != "no_control":
        mj.set_mjcb_control(control_callback)

    match mode:
        case "launcher":
            # Interactive window. Great for seeing what your robot does,
            # useless inside an evolutionary loop.
            viewer.launch(model=model, data=data)
        case "simple":
            # Headless. THIS is the one your EA uses.
            simple_runner(model, data, duration=SIM_DURATION)
        case "video":
            # Render to an mp4 - for the figures in your report.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(
                model,
                data,
                duration=SIM_DURATION,
                video_recorder=recorder,
            )
        case "frame":
            # A single image of the scene. Useful to check your spawn position
            # and that the robot is not clipping through the floor.
            single_frame_renderer(model, data, steps=1, show=True)
        case "no_control":
            # No controller attached: drag the hinges around by hand.
            viewer.launch(model=model, data=data)

    # Detach the callback again so the next run starts clean.
    mj.set_mjcb_control(None)

    # --- Score -------------------------------------------------------------- #
    final_position = get_core_position(data)
    max_displacement = max(max_displacement, _distance(final_position, initial_position))
    final_distance = _distance(final_position, target)
    end_time = float(data.time)

    elapsed_time = end_time - start_time
    remaining_dt = end_time - last_time
    if remaining_dt > 0.0:
        distance_integral += (
            0.5
            * (last_distance + final_distance)
            * remaining_dt
        )

    if elapsed_time > 0.0:
        mean_distance = distance_integral / elapsed_time
    else:
        mean_distance = final_distance

    if (time_to_target is None and final_distance <= TARGET_RADIUS):
        time_to_target = elapsed_time

    fitness = fitness_function(
        initial_position = initial_position,
        final_position = final_position,
        target_position=target,
        mean_distance=mean_distance,
        time_to_target=time_to_target,
        duration=elapsed_time,
    )

    fitness += STILL_PENALTY * max(0.0, 1.0 - max_displacement / STILL_RADIUS)

    console.log(f"start  : {np.round(initial_position, 3)}")
    console.log(f"end    : {np.round(final_position, 3)}")
    console.log(f"target : {np.round(TARGET_POSITION, 3)}")
    console.log(f"fitness: {fitness:.4f}   (lower is better)")

    console.log(f"final distance: {final_distance:.4f}")
    console.log(f"mean distance: {mean_distance:.4f}")
    console.log(f"time to target: {time_to_target}")
    return fitness

POPULATION_SIZE = 100
GENERATIONS = 100
OFFSPRING_SIZE = 50
TOURNAMENT_SIZE = 5

MUTATION_RATE = 0.4
MUTATION_SIGMA = 0.25
WEIGHT_LIMIT = 3.0

CROSSOVER_PROBABILITY = 0.6
SWAP_PROBABILITY = 0.225

def make_individual(input_size: int, output_size: int) -> Individual:
    individual = Individual()
    individual.genotype = make_random_genome(input_size, output_size)
    return individual

def make_random_genome(input_size: int, output_size: int) -> list[float]:
    number_of_weights = (input_size * HIDDEN_SIZE + HIDDEN_SIZE * output_size)
    return RNG.normal(loc=0.0, scale=0.5, size=number_of_weights).tolist()

def decode_genome(genome: list[float], input_size: int, output_size: int) -> list[npt.NDArray[np.float64]]:
    genome_array = np.asarray(genome, dtype=np.float64)
    w1_size = input_size * HIDDEN_SIZE

    w1 = genome_array[:w1_size].reshape(input_size, HIDDEN_SIZE)
    w2 = genome_array[w1_size:].reshape(HIDDEN_SIZE, output_size)

    return [w1, w2]

def evaluate(population: Population) -> Population:
    unevaluated = list(population.unevaluated)

    for idx, individual in enumerate(unevaluated, start=1):
        console.log(f"Evaluating individual {idx}/{len(unevaluated)}")
        genome = cast(list[float], individual.genotype)

        individual.fitness = run_experiment(genome, mode="simple")
        console.log(f"fitness={individual.fitness:.4f}")
    return population

def parent_selection(population: Population) -> Population:
    alive = list(population.alive)
    for individual in alive:
        individual.tags = {"mating_count": 0}
    tournament_size = min(TOURNAMENT_SIZE, len(alive))

    for _ in range(OFFSPRING_SIZE):
        candidate_indices = RNG.choice(len(alive), size=tournament_size, replace=False)
        candidates = [
            alive[int(index)]
            for index in candidate_indices
        ]

        winner = min(candidates, key=lambda ind: ind.fitness)
        mating_count = winner.tags.get("mating_count", 0)
        winner.tags = {
            "mating_count": mating_count + 1
        }
    return population

def survivor_selection(population: Population) -> Population:
    ranked = sorted(population.alive, key=lambda ind: ind.fitness)
    for individual in ranked[POPULATION_SIZE:]:
        individual.alive = False
    return population

def crossover(population: Population) -> Population:
    parent_slots: list[Individual] = []

    for parent in population.alive:
        mating_count = int(parent.tags.get("mating_count", 0))
        parent_slots.extend([parent] * mating_count)

    order = RNG.permutation(len(parent_slots))
    parent_slots = [
        parent_slots[int(idx)]
        for idx in order
    ]

    for idx in range (0, len(parent_slots), 2):
        parent_a = parent_slots[idx]
        parent_b = parent_slots[idx + 1]

        if RNG.random() < CROSSOVER_PROBABILITY:
            genome_a, genome_b = Crossover.uniform(
                parent_a.genotype,
                parent_b.genotype,
                swap_probability=SWAP_PROBABILITY
            )
        else:
            genome_a = cast(list[float], parent_a.genotype).copy()
            genome_b = cast(list[float], parent_b.genotype).copy()

        child_a = Individual()
        child_a.genotype = genome_a
        child_b = Individual()
        child_b.genotype = genome_b
        population.extend([child_a, child_b])
    return population

def mutate(population: Population, variation=False, variation_probability: float = 0.25) -> Population:
    if variation:
        return mutate_variation(population, variation_probability)
    else:
        return mutate_normal(population)

def mutate_normal(population: Population) -> Population:
    for individual in population.unevaluated:
        genome = np.asarray(
            cast(list[float], individual.genotype),
            dtype=np.float64
        ).copy()

        mutation_mask = RNG.random(genome.size) < MUTATION_RATE
        number_of_mutations = int(np.count_nonzero(mutation_mask))

        genome[mutation_mask] += RNG.normal(
            loc=0.0,
            scale=MUTATION_SIGMA,
            size=number_of_mutations
        )

        genome = np.clip(genome, -WEIGHT_LIMIT, WEIGHT_LIMIT)
        individual.genotype = genome.tolist()
    return population

def mutate_variation(population: Population, variation_probability: float = 0.25) -> Population:
    for individual in population.unevaluated:
        genome = np.asarray(
            cast(list[float], individual.genotype),
            dtype=np.float64
        ).copy()

        mutation_mask = RNG.random(genome.size) < variation_probability
        number_of_mutations = int(np.count_nonzero(mutation_mask))

        genome[mutation_mask] += RNG.normal(
            loc=0.0,
            scale=MUTATION_SIGMA,
            size=number_of_mutations
        )

        genome = np.clip(genome, -WEIGHT_LIMIT, WEIGHT_LIMIT)
        individual.genotype = genome.tolist()
    return population

def save_fitness_csv(history: list[dict[str, int | float]]) -> None:
    FITNESS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["generation", "best", "mean", "worst", "std"]
    with FITNESS_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

def plot_fitness_from_csv() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.size": 14})
    with FITNESS_CSV.open(newline="", encoding="utf-8") as source:
        history = list(csv.DictReader(source))
    generations = [int(row["generation"]) for row in history]
    figure, axes = plt.subplots(2,1, figsize=(9,7), sharex=True)
    for axis, key, title in (
        (axes[0], "mean", "Mean fitness of the population"),
        (axes[1], "best", "Best fitness of the population"),
    ):
        axis.plot(generations, [float(row[key]) for row in history], linewidth=2)
        axis.set(title=title, ylabel="Fitness (lower is better)")
        axis.grid(alpha=0.25)
    axes[1].set_xlabel("Generation (0 = Start population)")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    figure.tight_layout()
    figure.savefig(DATA / f"fitness_seed_{SEED}.png", dpi=150)
    plt.close(figure)

VARIATION_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPERIMENT_SEEDS = (42, 7, 1124, 8486, 2026)

def run_mutation_experiment(values=VARIATION_VALUES, seeds=EXPERIMENT_SEEDS) -> None:
    experiment_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    experiment_dir = (
        CWD / "__data__" / SCRIPT_NAME / "mutation_experiments"
        / f"experiment_{experiment_id}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=False)

    for probability in values:
        for seed in seeds:
            try:
                main(
                    seed=seed,
                    render_best=True,
                    variation=True,
                    variation_probability=probability,
                    output_dir=(
                        experiment_dir
                        / f"mutation_{probability}"
                        / f"seed_{seed}"
                    )
                )
            finally:
                mj.set_mjcb_control(None)

def main(seed: int = 42, render_best: bool = True, variation: bool = False, variation_probability: float = 0.25, output_dir: Path | None = None) -> float:
    global SEED, RNG, DATA, FITNESS_CSV

    SEED = seed
    RNG = np.random.default_rng(seed)
    set_seed(seed)

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M_%S_%f")
    DATA = (
        Path(output_dir)
        if output_dir is not None
        else CWD / "__data__" / SCRIPT_NAME / f"seed_{seed}_{run_id}"
    )
    FITNESS_CSV = DATA / "fitness_data" / "fitness.csv"

    # A quick look at the size of the problem you are about to search.
    DATA.mkdir(parents=True, exist_ok=False)

    mj.set_mjcb_control(None)
    world = build_world()
    robot = build_robot()
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mj.MjData(model)

    input_size = len(data.qpos)
    output_size = model.nu
    num_weights = (
        input_size * HIDDEN_SIZE
        + HIDDEN_SIZE * output_size
    )
    console.log(f"controller inputs (len(data.qpos)) : {input_size}")
    console.log(f"controller outputs (model.nu)      : {output_size}")
    console.log(f"genotype length (total weights)    : {num_weights}")

    config.target_population_size = POPULATION_SIZE

    initial_population = Population([
        make_individual(input_size, output_size)
        for _ in range(POPULATION_SIZE)
    ])

    history: list[dict[str, int | float]] = []
    def record_generation(population: Population) -> Population:
        fitness = np.asarray([ind.fitness for ind in population.alive], dtype=float)
        history.append({
            "generation": len(history),
            "best": float(fitness.min()),
            "mean": float(fitness.mean()),
            "worst": float(fitness.max()),
            "std": float(fitness.std())
        })
        return population

    initial_population = record_generation(evaluate(initial_population))

    def mutate_for_run(population: Population) -> Population:
        return mutate(
            population,
            variation=variation,
            variation_probability=variation_probability)

    operations: list[EAOperation] = [
        EAOperation(parent_selection),
        EAOperation(crossover),
        EAOperation(mutate_for_run),
        EAOperation(evaluate),
        EAOperation(survivor_selection),
        EAOperation(record_generation)
    ]

    ea = EA(
        initial_population,
        operations,
        num_steps=GENERATIONS,
        is_maximisation=False,
        db_file_path=DATA / "database.db",
        db_handling="halt"
    )
    ea.run()
    save_fitness_csv(history)
    best_individual = ea.get_solution("best", only_alive=False)

    if render_best:
        plot_fitness_from_csv()
        best_genome = cast(list[float], best_individual.genotype)
        run_experiment(best_genome, mode="video")

    return float(best_individual.fitness)

def optimize(n_trials: int = 20):
    import optuna

    def objective(trial):
        global MUTATION_RATE, MUTATION_SIGMA, TOURNAMENT_SIZE
        global CROSSOVER_PROBABILITY, SWAP_PROBABILITY

        MUTATION_RATE = trial.suggest_float(
            "MUTATION_RATE", 0.10,0.90
        )

        MUTATION_SIGMA = trial.suggest_float(
            "MUTATION_SIGMA", 0.005, 0.30, log=True
        )

        TOURNAMENT_SIZE = trial.suggest_int(
            "TOURNAMENT_SIZE", 2, 6
        )

        CROSSOVER_PROBABILITY = trial.suggest_float(
            "CROSSOVER_PROBABILITY", 0.0, 1.0
        )

        SWAP_PROBABILITY = trial.suggest_float(
            "SWAP_PROBABILITY", 0.10, 0.90
        )

        scores = []
        for seed in (1124, 8487, 7):
            try:
                scores.append(main(seed=seed, render_best=True))
            finally:
                mj.set_mjcb_control(None)

        trial.set_user_attr("seed_scores", scores)
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    globals().update(study.best_params)

    print("Best parameters: ", study.best_params)
    print("Best mean fitness: ", study.best_value)

    return study

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only",
        help="Recreate the plot from the stored fitness CSV without running the EA.",
        type=Path,
        metavar="CSV"
    )
    parser.add_argument("--optuna", action="store_true")
    parser.add_argument("--trials", type=int, default=20)

    parser.add_argument("--mutation-experiments", action="store_true")
    parser.add_argument(
        "--experiment-values", type=float, nargs="+", default=VARIATION_VALUES,
    )
    parser.add_argument(
        "--experiment-seeds", type=int, nargs="+", default=EXPERIMENT_SEEDS,
    )

    args = parser.parse_args()
    if args.plot_only is not None:
        FITNESS_CSV = args.plot_only
        DATA = FITNESS_CSV.parent.parent
        plot_fitness_from_csv()
    elif args.optuna:
        optimize(n_trials=args.trials)
    elif args.mutation_experiments:
        run_mutation_experiment(
            values=args.experiment_values,
            seeds=args.experiment_seeds
        )
    else:
        main()

# ============================================================================ #
#  YOUR JOB
# ============================================================================ #
#
# Everything above runs one robot with random weights. It will score badly, and
# it will score badly in a slightly different way every time you change SEED.
# Your task is to replace "random" with "evolved".
#
# Build a proper EA on top of `ariel.ec`. You are expected to use that module -
# it gives you the population/individual data model, the operators, and free
# persistence of every generation to a SQLite database, which you will want
# when it is time to plot convergence curves for the report.
#
#     from ariel.ec import EA, EAOperation, Individual, Population
#
# For a complete, runnable example of how those pieces fit together (a one-max
# EA with parent selection, crossover, mutation and survivor selection written
# as separate steps), read:
#
#     examples/new_EC_engine_example.py
#
# and the API documentation at:
#
#     https://ci-group.github.io/ariel/
#
# ---- EXPERIMENTAL RIGOUR ---------------------------------------------------
#
#   One run proves nothing. Repeat every configuration over several
#     independent seeds and report mean and spread.
#   Log best/mean/worst fitness per generation. The database `ariel.ec`
#     writes makes this straightforward.
#   Compare against a baseline. Random search with the same evaluation
#     budget is a simple, but reasonable choice; and it is nearly free to run.
#   Keep body, world, SIM_DURATION and fitness function identical across
#     everything you compare. Change one thing at a time.
#
# ============================================================================ #
