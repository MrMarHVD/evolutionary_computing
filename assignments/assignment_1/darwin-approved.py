"""EC A1 template code - evolving robot morphologies with ARIEL.

WHAT THIS FILE IS
-----------------
A *demo* file for starting you out with assignment 1.
It samples one body at random, decodes it, scores it
against a set of target bodies, and shows you the result.

*Your Job* section at the bottom of this file summarises the programming task. Full assignment description can be found in the pdf file on Canvas.


THE ASSIGNMENT IN A NUTSHELL
------------------------------
Evolve a robot BODY that is as structurally close as possible to a whole set
of given target bodies at once.

    fitness = mean tree edit distance to every body in TARGET_DIR,
              plus one standard deviation across those per-target distances
"""


# Standard library
import random
from copy import deepcopy
from pathlib import Path
from typing import Literal, cast

# Third-party libraries
import mujoco as mj
import networkx as nx
import numpy as np
import numpy.typing as npt
import torch
from mujoco import viewer

from ariel.ec import Individual, Population, Crossover, config, EAOperation, EA
# Local scripts
from tree_edit_distance import (
    distances_to_targets,
    mean_plus_std_tree_edit_distance,
    tree_edit_distance,
)

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.video_recorder import VideoRecorder

# Type aliases
type ViewerTypes = Literal["launcher", "video", "frame", "none"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
# NOTE: the tree operators use the `random` module, the NDE uses numpy for its
# own genotype vectors AND is a torch.nn.Module for its internal network - that
# network's weight initialisation uses torch's own RNG, entirely separate from
# numpy/random. If you're using "nde", seed all THREE or your runs will not be
# reproducible across separate script runs, even with the same seed value.
SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
HERE = Path(__file__).parent
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
TARGET_DIR: Path = HERE / "target_bodies"  # the bodies you must approach
NUM_OF_MODULES: int = 20  # module budget per evolved body
GENOTYPE: str = "nde"  # "nde" | "tree"
MODE: ViewerTypes = "frame"  # see show_body() for the options
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]

NUM_CHROMOSOMES = 3
INIT_MIN, INIT_MAX = -1.0, 1.0

POPULATION_SIZE = 200
OFFSPRING_SIZE = 200
NUM_GENERATIONS = 200
MUTATION_PROBABILITY = 0.1
MUTATION_SD = 0.1
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.6
NUM_CROSSOVER_POINTS = 4

# K-values to compare between the two conditions
MUTATIONS_PER_GENOTYPE = 5
#K = [0, 5, 10, 15, 20, 25, 30, 35, 40]

# ============================================================================ #
#  1. THE TARGET BODIES
# ============================================================================ #
#
# The targets are plain nx.DiGraph JSON files.
# They vary in size on purpose. A body that just matches the average module
# count will not score well against all of them.
#
# ============================================================================ #

def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """Load every target body graph from a directory.

    Returns
    -------
    list of nx.DiGraph
        One graph per JSON file, sorted by filename.

    Raises
    ------
    FileNotFoundError
        If the directory holds no target JSON files.
    """
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]

# ============================================================================ #
#  2. THE GENOTYPE CONTRACT
# ============================================================================ #
#
# You may use EITHER of ARIEL's two body encodings below. You may NOT invent
# your own, and CPPN is not offered for this assignment.
# Whichever you pick, the contract is the same and it is very short:
#
#       your genotype  --(its decoder)-->  nx.DiGraph  -->  fitness
#
# That DiGraph is the phenotype, and it is all the fitness function ever sees:
#
#       nodes carry   type      : "CORE" | "BRICK" | "HINGE"
#                     rotation  : "DEG_0" | "DEG_45" | "DEG_90"
#       edges carry   face      : "FRONT" | "BACK" | "RIGHT" | "LEFT"
#                                 | "TOP" | "BOTTOM"
#
# THE TWO ENCODINGS
#
#   "nde"   NeuralDevelopmentalEncoding + HighProbabilityDecoder
#           Genotype: three fixed-length float vectors (type / connection /
#           rotation genes). An INDIRECT encoding - a small vector is expanded
#           by a fixed neural network into probability matrices, which are
#           then decoded greedily into a body.
#           -> Fixed-length real vector. Standard real-valued operators work
#              out of the box. But the genotype-phenotype map is wildly
#              non-linear: a small mutation can rebuild the robot entirely.
#           -> IMPORTANT: `NeuralDevelopmentalEncoding`'s internal network is
#              randomly (re-)initialised every time you construct it, and NOT
#              derived from the genotype you pass in. If your EA's decode step
#              builds a fresh `NeuralDevelopmentalEncoding(...)` per individual
#              (the natural way to write it - see `random_nde_body` below),
#              the SAME genotype decodes to a DIFFERENT random body every call,
#              and fitness stops reflecting the genotype at all. Construct it
#              ONCE for your whole run and reuse that one instance's
#              `.forward()` for every genotype you decode.
#           -> ALSO IMPORTANT: `NeuralDevelopmentalEncoding` is a
#              `torch.nn.Module`. Its weight initialisation uses torch's own
#              RNG, entirely separate from numpy/random. `np.random.seed(...)`
#              and `random.seed(...)` do NOT control it - you also need
#              `torch.manual_seed(...)`, or your results will not reproduce
#              across separate runs even with "the same" seed.
#
# Below, each encoding gets ONE random genotype, decoded to a graph. That is
# your starting point, not your solution: your EA has to search this space,
# not sample it once.
#
# ============================================================================ #

# NDE settings
GENOTYPE_SIZE: int = 64  # length of each of the three NDE gene vectors

# Constructed ONCE, at import time, and reused for every decode call below and
# in your own EA. See the "IMPORTANT" note on "nde" in THE GENOTYPE CONTRACT
# above: rebuilding this per individual silently breaks the genotype -> body
# mapping, because its internal network randomises on construction.
_NDE = NeuralDevelopmentalEncoding(
    number_of_modules=NUM_OF_MODULES,
    genotype_size=GENOTYPE_SIZE,
)

# ============================================================================ #
#  3. FITNESS
# ============================================================================ #
#
# Fitness is the MEAN tree edit distance to every target body, PLUS one
# standard deviation across those per-target distances. LOWER IS BETTER, and
# 0.0 would mean your body is identical to all of them at once - which, since
# the targets differ from each other, is impossible. There is a floor above
# zero here and you will not reach it. Work out roughly where it is: a body
# cannot be closer to a set than the set is to itself.
#
# The distance itself lives in tree_edit_distance.py.
# Read that file - you cannot reason about your EA's behaviour without knowing what it is climbing.
#
# ============================================================================ #

def fitness_function(
    body: nx.DiGraph,
    targets: list[nx.DiGraph],
) -> float:
    """Score one body against the whole target set. LOWER IS BETTER.

    Some things worth thinking about:
      * The std term charges for unevenness - body that is mediocre against every target
        and one that is excellent on most but bad on one can still land close
        in fitness, but the latter is penalized a bit more.
      * Nothing here rewards small bodies. Does your EA bloat? Should a size
        penalty be part of fitness, or is that the encoding's job?
    """
    return mean_plus_std_tree_edit_distance(body, targets)


# ============================================================================ #
#  4. LOOKING AT A BODY
# ============================================================================ #


def show_body(
    body: nx.DiGraph,
    mode: ViewerTypes = MODE,
    file_name: str = "body",
) -> None:
    """Build a body graph in MuJoCo and look at it.

    There is no controller and no physics worth speaking of - this exists so
    you can SEE what your fitness function is actually rewarding. Do this
    early and often. A number going down is not evidence that the bodies look
    anything like the targets.
    """
    if mode == "none":
        return

    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    world = SimpleFlatWorld()
    robot = construct_mjspec_from_graph(body)
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    match mode:
        case "launcher":
            # Interactive window. Drag the modules around; nothing drives them.
            viewer.launch(model=model, data=data)
        case "frame":
            # A still image - the cheapest way to eyeball a body.
            save_path = str(DATA / f"{file_name}.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
            console.log(f"saved {save_path}")
        case "video":
            # Mostly useful for showing a body slumping under gravity.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(model, data, duration=5.0, video_recorder=recorder)

# ============================================================================ #
#  EA functions
#           ---- Best body solution ----                                                                                                          darwin-approved.py:435
#           individual ID : 1136                                                                                                                  darwin-approved.py:436
#           modules       : 13 modules                                                                                                            darwin-approved.py:437
#           per-target    : 10.0, 9.0, 13.0, 12.5, 14.5                                                                                           darwin-approved.py:438
#           fitness       : 13.8149   (lower is better)
# ============================================================================ #

def decode_individual(individual: Individual) -> nx.DiGraph:
    raw_genotype = cast(list[list[float]], individual.genotype)
    nde_genotype: list[npt.NDArray[np.float32]] = [
        np.asarray(chromosome, np.float32)
        for chromosome in raw_genotype
    ]
    type_p, conn_p, rot_p = _NDE.forward(nde_genotype)

    decoder = HighProbabilityDecoder(NUM_OF_MODULES)
    return decoder.probability_matrices_to_graph(
        type_p,
        conn_p,
        rot_p)

def make_individual() -> Individual:
    indi = Individual()
    geno = [
        RNG.uniform(INIT_MIN, INIT_MAX, GENOTYPE_SIZE)
        .astype(np.float32)
        .tolist()
        for _ in range(NUM_CHROMOSOMES)
    ]

    indi.genotype = geno
    return indi

def mutate(population: Population) -> Population:
    #if RNG.random() < MUTATION_PROBABILITY:
        for ind in population.unevaluated:
            if RNG.random() < MUTATION_PROBABILITY:
                break
            raw_g = cast(list[list[float]], ind.genotype)
            genotype = np.asarray(raw_g, dtype=np.float32).copy()

            flat_genotype = genotype.reshape(-1)

            mutation_positions = RNG.choice(
                genotype.size,
                size=MUTATIONS_PER_GENOTYPE,
                replace=False
            )
            new_values = RNG.uniform(
                INIT_MIN,
                INIT_MAX,
                size=MUTATIONS_PER_GENOTYPE
            ).astype(np.float32)

            flat_genotype[mutation_positions] = new_values

            ind.genotype = genotype.tolist()
            ind.fitness_ = None
            ind.requires_eval = True
        return population

# Copied/adapted from the example new_EC_engine_example.py
def crossover(population: Population) -> Population:
    parents_slots: list[Individual] = []

    for ind in population.alive:
        mating_count = ind.tags.get("mating_count", 0)
        parents_slots.extend([ind] * mating_count)
    random.shuffle(parents_slots)

    for idx in range(0, len(parents_slots), 2):
        parent_a = parents_slots[idx]
        parent_b = parents_slots[idx + 1]

        if RNG.random() < CROSSOVER_PROBABILITY:
            g_a, g_b = Crossover.n_point(
                parent_a.genotype,
                parent_b.genotype,
                NUM_CROSSOVER_POINTS
            )
        else:
            g_a = deepcopy(parent_a.genotype)
            g_b = deepcopy(parent_b.genotype)

        child_a = Individual()
        child_a.genotype = g_a

        child_b = Individual()
        child_b.genotype = g_b

        population.extend([child_a, child_b])
    return population

# Copied from the example new_EC_engine_example.py
def survivor_selection(population: Population) -> Population:
    shuffled = population.alive.shuffle()
    alive_count = len(shuffled)
    for idx in range(0, len(shuffled) - 1, 2):
        if alive_count <= config.target_population_size:
            break
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]
        if ind_a.fitness_ <= ind_b.fitness_:
            ind_b.alive = False
        else:
            ind_a.alive = False
        alive_count -= 1
    return population

def parent_selection(population: Population) -> Population:
    alive_ind = list(population.alive)
    for ind in alive_ind:
        ind.tags = {"mating_count": 0}

    for _ in range(OFFSPRING_SIZE):
        candidates = random.sample(
            alive_ind,
            k=TOURNAMENT_SIZE
        )
        winner = min(candidates, key=get_fitness)

        current_count = winner.tags.get("mating_count", 0)
        winner.tags = {"mating_count": current_count + 1}
    return population

def get_fitness(ind: Individual) -> float:
    return ind.fitness

def evaluate(population: Population, targets: list[nx.DiGraph]) -> Population:
    for ind in population.unevaluated:
        body = decode_individual(ind)
        ind.fitness = fitness_function(body, targets)
    return population

# ============================================================================ #
#  5. ENTRY POINT
# ============================================================================ #

def main() -> None:
    """Run the evolutionary algorithm and report its best solution"""
    targets = load_targets()
    config.target_population_size = POPULATION_SIZE

    console.log(f"encoding      : {GENOTYPE}")
    console.log(f"module budget : {NUM_OF_MODULES}")
    console.log(f"population    : {POPULATION_SIZE}")
    console.log(f"generations   : {NUM_GENERATIONS}")
    console.log(f"seed          : {SEED}")
    console.log(f"targets       : {len(targets)} bodies from {TARGET_DIR.name}")
    console.log(
        "target sizes  : "
        + ", ".join(str(t.number_of_nodes()) for t in targets),
    )

    # How far apart are the targets from each other? Your fitness cannot go
    # below the best possible compromise, and this is the clue to where that is.
    spread = [
        tree_edit_distance(a, b)
        for i, a in enumerate(targets)
        for b in targets[i + 1:]
    ]
    console.log(f"target spread : mean pairwise distance {np.mean(spread):.2f}")

    initial = Population([make_individual() for _ in range(POPULATION_SIZE)])
    initial = evaluate(initial, targets)

    ops: list[EAOperation] = [
        EAOperation(parent_selection),
        EAOperation(crossover),
        EAOperation(mutate),
        EAOperation(evaluate, targets=targets),
        EAOperation(survivor_selection),
    ]

    ea = EA(
        initial,
        ops,
        num_steps=NUM_GENERATIONS,
        is_maximisation=False # Whether higher fitness is better -> Smaller fitness is better in our case
    )
    ea.run()

    best_individual = ea.get_solution("best", only_alive=False) # TODO: Should this parameter be true or false?
    best_body = decode_individual(best_individual)
    per_target_distances = list(distances_to_targets(best_body, targets))

    console.log("")
    console.log("---- Best body solution ----")
    console.log(f"individual ID : {best_individual.id}")
    console.log(f"modules       : {best_body.number_of_nodes()} modules")
    console.log(
        "per-target    : "
        + ", ".join(f"{d:.1f}" for d in per_target_distances),
    )
    console.log(f"fitness       : {best_individual.fitness:.4f}   (lower is better)")

    show_body(best_body, MODE, file_name=f"best_body_{GENOTYPE}_seed_{SEED}")


if __name__ == "__main__":
    main()

# ============================================================================ #
#  YOUR JOB
# ============================================================================ #
#
# Everything above samples ONE body at random and scores it. Your task is to
# replace "random" with "evolved".
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
# For morphology-specific evolution with the tree encoding, read:
#
#     examples/c_genotypes/1_body_evolution_tree.py
#
# and the API documentation at:
#
#     https://ci-group.github.io/ariel/
#
# ---- GENOTYPE - DEPENDENT "GOTCHA"S -------------------------
#   NDE: REPRODUCIBILITY   If you're using "nde": construct
#     `NeuralDevelopmentalEncoding` ONCE for your whole run, never per
#     individual or per generation, AND call `torch.manual_seed(...)` in
#     addition to the numpy/random seeds. See the two "IMPORTANT" notes under
#     "nde" in THE GENOTYPE CONTRACT above - getting either wrong means your
#     your runs won't reproduce cleanly.
#
