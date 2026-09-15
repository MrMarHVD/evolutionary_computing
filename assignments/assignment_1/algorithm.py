"""Compare how many NDE genes are mutated when evolving robot bodies."""

from __future__ import annotations

# Standard library
import random
from pathlib import Path

# Third-party libraries
import networkx as nx
import numpy as np
import torch

# ARIEL
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec import (
    EA,
    EAOperation,
    FloatMutator,
    FloatsGenerator,
    Crossover,
    Individual,
    Population,
    config,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding

# Assignment fitness functions
from tree_edit_distance import (
    distances_to_targets,
    mean_plus_std_tree_edit_distance,
)


# 1. Settings
HERE = Path(__file__).resolve().parent
TARGET_DIR = HERE / "target_bodies"
GENOTYPE_SIZE = 64
NUM_CHROMOSOMES = 3
INIT_MIN, INIT_MAX = -1.0, 1.0
NUM_MODULES = 20
# TODO: Set population size, generations, seeds, k values, and operator settings.
# k counts distinct positions across all three chromosomes (192 genes total).

# Constants
POPULATION_SIZE = 1000 # TODO: determine the right size, use this for both initial population size and number of offspring per generation
NUM_GENERATIONS = 100
MUTATION_PROBABILITY = 1.0
MUTATION_SD = 0.1
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.5
SEEDS = list(range(5))


# K-values to compare between the two conditions
K = [10, 20]

# 2. Load target body graphs
# TODO: Load the target JSON files in sorted order.


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
    """Run the experiment once its operations are implemented."""
    pass


if __name__ == "__main__":
    main()
