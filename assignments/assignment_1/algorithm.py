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
from ariel.ec.generators import _rng as ariel_rng
from ariel.ec.individual import JSONIterable

# Settings and constants
HERE = Path(__file__).resolve().parent
TARGET_DIR = HERE / "target_bodies"
RESULTS_DIR = HERE / "results"
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


def seed_operators(seed: int) -> np.random.Generator:
    """Reset operator randomness and return an RNG for mutation positions."""
    random.seed(seed)
    # Update in place: crossover holds a reference to this same RNG object.
    # ARIEL currently provides no public operator-seeding function.
    ariel_rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
    return np.random.default_rng(seed)


# Create an NDE network from a seed
def create_nde(seed: int) -> NeuralDevelopmentalEncoding:
    torch.manual_seed(seed)

    return NeuralDevelopmentalEncoding(
        number_of_modules=NUM_MODULES,
        genotype_size=GENOTYPE_SIZE,
    )


# Initialise the population
def initialise_pop(seed: int) -> Population:
    rng = np.random.default_rng(seed)
    individuals: list[Individual] = []

    for i in range(POPULATION_SIZE):
        individual = Individual()
        individual.genotype = rng.uniform(
            low=INIT_MIN,
            high=INIT_MAX,
            size=(NUM_CHROMOSOMES, GENOTYPE_SIZE),
        ).tolist()

        individuals.append(individual)

    return Population(individuals)


def decode_genome(
    genotype: JSONIterable,
    nde: NeuralDevelopmentalEncoding,
) -> nx.DiGraph[int]:
    """Decode an NDE genome into the body graph used for fitness."""
    genes = np.asarray(genotype, dtype=np.float32)
    if genes.shape != (NUM_CHROMOSOMES, GENOTYPE_SIZE):
        raise ValueError(
            f"Expected genome shape {(NUM_CHROMOSOMES, GENOTYPE_SIZE)}, "
            f"got {genes.shape}"
        )
    scores = nde.forward(list(genes))
    decoder = HighProbabilityDecoder(NUM_MODULES)
    return decoder.probability_matrices_to_graph(*scores)


def evaluate_population(
    population: Population,
    nde: NeuralDevelopmentalEncoding,
    targets: list[nx.DiGraph[int]],
) -> Population:
    """Assign distance-based fitness to individuals requiring evaluation."""
    if not targets:
        raise ValueError("At least one target body is required")
    for individual in population.unevaluated:
        body = decode_genome(individual.genotype, nde)
        # The fitness setter also clears requires_eval.
        individual.fitness = mean_plus_std_tree_edit_distance(body, targets)
    return population


"""RUN THE ACTUAL ALGORITHM"""
# 3. Initialize the population and fixed NDE network
# TODO: Generate genomes and create one NDE network per repetition.
# Reuse the initial genomes and network weights across k conditions.
# Seed Python, PyTorch, and both our own and ARIEL's operator RNGs explicitly.


#
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

    for seed in SEEDS:
        # create network
        nde = create_nde(seed)
        # generate population
        for k in K:
            population = initialise_pop(seed)
        #

    """Score one randomly-sampled body against the target set."""

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
