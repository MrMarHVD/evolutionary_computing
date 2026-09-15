"""Compare how many NDE genes are mutated when evolving robot bodies."""

from __future__ import annotations

# Standard library
import random
from pathlib import Path

# Third-party libraries
import networkx as nx
import numpy as np
import torch
# Assignment fitness functions
from tree_edit_distance import (distances_to_targets,
                                mean_plus_std_tree_edit_distance)

# ARIEL
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import \
    load_graph_from_json
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import \
    HighProbabilityDecoder
from ariel.ec import (EA, Crossover, EAOperation, FloatMutator,
                      FloatsGenerator, Individual, Population, config)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding

# Settings and constants
HERE = Path(__file__).resolve().parent
TARGET_DIR = HERE / "target_bodies"
GENOTYPE_SIZE = 64
NUM_CHROMOSOMES = 3
INIT_MIN, INIT_MAX = -1.0, 1.0
NUM_MODULES = 20

POPULATION_SIZE = 1000  # TODO: determine the right size, use this for both initial population size and number of offspring per generation
NUM_GENERATIONS = 100
MUTATION_PROBABILITY = 1.0
MUTATION_SD = 0.1
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.5
SEEDS = list(range(5))


# K-values to compare between the two conditions
K = [0, 5, 10, 15, 20, 25, 30, 35, 40]

# Load target graphs from predefined target directory

"""DEFINE THE FUNCTIONS WE WILL USE"""


def get_targets() -> list[nx.DiGraph[int]]:
    paths = sorted(TARGET_DIR.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No graphs found in {TARGET_DIR}")

    return [load_graph_from_json(path) for path in paths]


def random_nde_body(num_modules: int = NUM_MODULES) -> nx.DiGraph:
    """Sample a random NDE genotype and decode it into a body graph.

    THIS IS THE FUNCTION YOUR EA REPLACES. The three vectors below are the
    genotype: that is what you mutate, recombine and select on. Note this
    function does NOT construct its own `NeuralDevelopmentalEncoding` - it
    reuses the module-level `_NDE` instance. Do the same in your EA.

    `num_modules` must match the value `_NDE` was built with (NUM_OF_MODULES).
    """
    genotype = [
        RNG.uniform(-1.0, 1.0, GENOTYPE_SIZE).astype(np.float32)  # module types
        for _ in range(3)  # types, connections, rotations
    ]

    type_p, conn_p, rot_p = _NDE.forward(genotype)

    decoder = HighProbabilityDecoder(num_modules)
    return decoder.probability_matrices_to_graph(type_p, conn_p, rot_p)


"""RUN THE ACTUAL ALGORITHM"""
# 3. Initialize the population and fixed NDE network
# TODO: Generate genomes and create one NDE network per repetition.
# Reuse the initial genomes and network weights across k conditions.
# Seed Python, PyTorch, and both our own and ARIEL's operator RNGs explicitly.


# 4. Decode genomes and evaluate fitness
# TODO: Decode to graphs and assign mean-plus-standard-deviation distance.
# Lower fitness is better; no physics simulation is needed.


# 5. Select parents
# TODO: Run tournaments on evaluated individuals, preferring lower fitness.


# 6. Create offspring through crossover
# TODO: Recombine parent genomes and store them in new Individuals.


# 7. Mutate exactly k gene positions
# TODO: Select k distinct positions and apply fixed-strength Gaussian noise.
# Keep the mutation-event probability constant across conditions.


# 8. Select survivors
# TODO: Keep the best individuals from parents plus evaluated offspring.


# 9. Run repetitions and record results
# TODO: Assemble EAOperation steps and use a separate database for each run.


def main() -> None:
    """Score one randomly-sampled body against the target set."""
    targets = load_targets()

    console.log(f"encoding      : {GENOTYPE}")
    console.log(f"module budget : {NUM_OF_MODULES}")
    console.log(f"targets       : {len(targets)} bodies from {TARGET_DIR.name}")
    console.log(
        "target sizes  : " + ", ".join(str(t.number_of_nodes()) for t in targets),
    )

    # How far apart are the targets from each other? Your fitness cannot go
    # below the best possible compromise, and this is the clue to where that is.
    spread = [
        tree_edit_distance(a, b)
        for i, a in enumerate(targets)
        for b in targets[i + 1 :]
    ]
    console.log(f"target spread : mean pairwise distance {np.mean(spread):.2f}")

    # --- One random body --------------------------------------------------- #
    body = random_body(GENOTYPE, NUM_OF_MODULES)
    fitness = fitness_function(body, targets)

    console.log("")
    console.log(f"random body   : {body.number_of_nodes()} modules")
    console.log(
        "per-target    : "
        + ", ".join(f"{d:.1f}" for d in distances_to_targets(body, targets)),
    )
    console.log(f"fitness       : {fitness:.4f}   (lower is better)")

    show_body(body, MODE, file_name=f"random_{GENOTYPE}")


if __name__ == "__main__":
    main()
