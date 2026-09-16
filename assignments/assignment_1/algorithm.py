"""Compare how many NDE genes are mutated when evolving robot bodies."""

from __future__ import annotations

import csv
import json
# Standard library
import random
from pathlib import Path
from typing import Any

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
from ariel.ec.generators import _rng as ariel_rng
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
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


def select_parents(population: Population) -> Population:
    """Select tournament winners to produce POPULATION_SIZE offspring."""
    candidates = population.alive
    parent_count = POPULATION_SIZE + POPULATION_SIZE % 2
    return Population(
        [
            candidates.sample(TOURNAMENT_SIZE).best(sort="min")[0]
            for _ in range(parent_count)
        ]
    )


def create_offspring(parents: Population) -> Population:
    """Cross paired parents uniformly, returning new, unevaluated individuals."""
    required_output = POPULATION_SIZE + POPULATION_SIZE % 2
    children: list[Individual] = []
    for index in range(0, required_output, 2):
        genomes = Crossover.uniform(
            parents[index].genotype,
            parents[index + 1].genotype,
            swap_probability=CROSSOVER_PROBABILITY,
        )
        for genome in genomes:
            child = Individual()
            child.genotype = genome
            children.append(child)
    return Population(children[:POPULATION_SIZE])


def mutate_k(
    genotype: JSONIterable,
    k: int,
    rng: np.random.Generator,
) -> list[list[float]]:
    """Perturb k distinct positions across the genome, without clipping."""
    genes = np.array(genotype, dtype=float, copy=True)
    if k:
        flat = genes.reshape(
            -1
        )  # Flatten so that mutation can be applied as though it were a single genome
        indices = rng.choice(flat.size, size=k, replace=False)
        flat[indices] = FloatMutator.gaussian(
            flat[indices].tolist(),
            std=MUTATION_SD,
            mutation_probability=1.0,
        )
    return genes.tolist()


def mutate_population(
    offspring: Population,
    k: int,
    rng: np.random.Generator,
) -> Population:
    """Apply the mutation-event to each offspring."""
    for individual in offspring:
        if rng.random() < MUTATION_PROBABILITY:
            individual.genotype = mutate_k(individual.genotype, k, rng)
            individual.requires_eval = True
            individual.fitness_ = None
    return offspring


def produce_offspring(
    population: Population,
    k: int,
    rng: np.random.Generator,
) -> Population:
    """Append mutated offspring while retaining parents for survivor selection."""
    offspring = create_offspring(select_parents(population))
    mutate_population(offspring, k, rng)
    population.extend(offspring)
    return population


def murder_majority(
    population: Population,
    size: int = POPULATION_SIZE,
) -> Population:
    """Mark all but the best living individuals dead; retain rows for the EA."""
    candidates = population.alive
    for individual in candidates.sort(sort="min")[size:]:
        individual.alive = False
    return population


def record_generation(
    population: Population,
    seed: int,
    k: int,
    history: list[dict[str, Any]],
) -> Population:
    """Append survivor statistics and raw fitness values; first record is g=0."""
    living = population.alive
    fitness = np.array([ind.fitness for ind in living], dtype=float)
    history.append(
        {
            "seed": seed,
            "k": k,
            "generation": len(history),
            "best": float(fitness.min()),
            "mean": float(fitness.mean()),
            "worst": float(fitness.max()),
            "std": float(fitness.std()),
            "fitness": fitness.tolist(),
        }
    )
    return population


def build_operations(
    nde: NeuralDevelopmentalEncoding,
    targets: list[nx.DiGraph[int]],
    seed: int,
    k: int,
    rng: np.random.Generator,
    history: list[dict[str, Any]],
) -> list[EAOperation]:
    """Build EA steps; evaluate and record the initial population beforehand."""
    # Explicit bound arguments also avoid ARIEL's unresolved-annotation check.
    return [
        EAOperation(produce_offspring, k=k, rng=rng),
        EAOperation(evaluate_population, nde=nde, targets=targets),
        EAOperation(murder_majority, size=POPULATION_SIZE),
        EAOperation(record_generation, seed=seed, k=k, history=history),
    ]


def get_run_directory(seed: int, k: int) -> Path:
    """Create the output directory for one seed and mutation condition."""
    directory = RESULTS_DIR / f"seed_{seed}" / f"k_{k}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_run_results(
    population: Population,
    nde: NeuralDevelopmentalEncoding,
    targets: list[nx.DiGraph[int]],
    seed: int,
    k: int,
    history: list[dict[str, Any]],
) -> Path:
    """Save settings, history, final genomes, and the best body and NDE weights."""
    directory = get_run_directory(seed, k)
    living = population.alive
    best = living.best(sort="min")[0]
    body = decode_genome(best.genotype, nde)
    settings = {
        "seed": seed,
        "k": k,
        "population_size": POPULATION_SIZE,
        "num_generations": NUM_GENERATIONS,
        "num_modules": NUM_MODULES,
        "genotype_size": GENOTYPE_SIZE,
        "num_chromosomes": NUM_CHROMOSOMES,
        "init_min": INIT_MIN,
        "init_max": INIT_MAX,
        "mutation_probability": MUTATION_PROBABILITY,
        "mutation_sd": MUTATION_SD,
        "tournament_size": TOURNAMENT_SIZE,
        "crossover_swap_probability": CROSSOVER_PROBABILITY,
        "fitness": "mean_plus_std_tree_edit_distance",
        "minimize": True,
    }
    payloads = {
        "settings.json": settings,
        "history.json": history,
        "final_population.json": [
            {"genotype": ind.genotype, "fitness": ind.fitness} for ind in living
        ],
        "best.json": {
            "genotype": best.genotype,
            "fitness": best.fitness,
            "distances_to_targets": list(distances_to_targets(body, targets)),
        },
        "best_body.json": nx.node_link_data(body, edges="edges"),
        "targets.json": [nx.node_link_data(t, edges="edges") for t in targets],
    }
    for name, payload in payloads.items():
        (directory / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save(nde.state_dict(), directory / "nde.pth")
    with (directory / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["seed", "k", "generation", "best", "mean", "worst", "std"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(history)
    return directory


def plot_run_results(history: list[dict[str, Any]], directory: Path) -> None:
    """Save convergence curves and initial/final fitness histograms."""
    from matplotlib.figure import Figure

    directory.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(8, 5))
    axis = figure.subplots()
    generations = [row["generation"] for row in history]
    for statistic in ("best", "mean", "worst"):
        axis.plot(generations, [row[statistic] for row in history], label=statistic)
    axis.set(xlabel="Generation", ylabel="Fitness (lower is better)")
    axis.legend()
    figure.savefig(directory / "convergence.png", bbox_inches="tight")

    figure = Figure(figsize=(8, 5))
    axis = figure.subplots()
    bins = np.histogram_bin_edges(
        history[0]["fitness"] + history[-1]["fitness"], bins="auto"
    )
    for label, row in (("Initial", history[0]), ("Final", history[-1])):
        axis.hist(row["fitness"], bins=bins, alpha=0.5, label=label)
    axis.set(xlabel="Fitness", ylabel="Number of individuals")
    axis.legend()
    figure.savefig(directory / "fitness_histogram.png", bbox_inches="tight")


def plot_results() -> None:
    """Compare saved conditions across SEEDS; require all configured runs."""
    from matplotlib.figure import Figure

    convergence = Figure(figsize=(9, 6))
    convergence_axis = convergence.subplots()
    histogram = Figure(figsize=(9, 6))
    histogram_axis = histogram.subplots()
    summary: list[dict[str, Any]] = []
    final_by_k: dict[int, list[float]] = {}
    for k in K:
        histories = [
            json.loads(
                (RESULTS_DIR / f"seed_{seed}" / f"k_{k}" / "history.json").read_text()
            )
            for seed in SEEDS
        ]
        values = np.array([[row["best"] for row in rows] for rows in histories])
        mean, std = values.mean(axis=0), values.std(axis=0)
        (line,) = convergence_axis.plot(generations, mean, label=f"k={k}")
        convergence_axis.fill_between(
            generations, mean - std, mean + std, color=line.get_color(), alpha=0.15
        )
        final_by_k[k] = values[:, -1].tolist()
        summary.append(
            {
                "k": k,
                "runs": len(histories),
                "mean_final_best": float(mean[-1]),
                "std_final_best": float(std[-1]),
            }
        )
    bins = np.histogram_bin_edges(
        [v for values in final_by_k.values() for v in values], bins="auto"
    )
    for k, values in final_by_k.items():
        histogram_axis.hist(values, bins=bins, histtype="step", label=f"k={k}")
    convergence_axis.set(
        xlabel="Generation", ylabel="Best fitness: mean ± standard deviation"
    )
    histogram_axis.set(xlabel="Final best fitness", ylabel="Number of repetitions")
    convergence_axis.legend()
    histogram_axis.legend()
    convergence.savefig(RESULTS_DIR / "comparison_convergence.png", bbox_inches="tight")
    histogram.savefig(RESULTS_DIR / "comparison_histogram.png", bbox_inches="tight")
    with (RESULTS_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["k", "runs", "mean_final_best", "std_final_best"]
        )
        writer.writeheader()
        writer.writerows(summary)


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
