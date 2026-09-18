"""Compare how many NDE genes are mutated when evolving robot bodies."""

from __future__ import annotations

import csv
import json
# Standard library
import random
import sys
from pathlib import Path
from typing import Any

# Allow this file to be run directly from the assignment directory.
ARIEL_SRC = Path(__file__).resolve().parents[2] / "src"
if str(ARIEL_SRC) not in sys.path:
    sys.path.insert(0, str(ARIEL_SRC))

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
from ariel.ec import Crossover, FloatMutator, Individual, Population
from ariel.ec.generators import _rng as ariel_rng
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.ec.individual import JSONIterable

# Settings and constants
HERE = Path(__file__).resolve().parent
TARGET_DIR = HERE / "target_bodies"
RESULTS_DIR = HERE / "results_100_5"
GENOTYPE_SIZE = 64
NUM_CHROMOSOMES = 3
INIT_MIN, INIT_MAX = -1.0, 1.0
NUM_MODULES = 20

POPULATION_SIZE = 100  # TODO: determine the right size, use this for both initial population size and number of offspring per generation
NUM_GENERATIONS = 100
MUTATION_PROBABILITY = 0.1
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
    scores = nde.forward(list(genes))
    decoder = HighProbabilityDecoder(NUM_MODULES)
    return decoder.probability_matrices_to_graph(*scores)


def evaluate_population(
    population: Population,
    nde: NeuralDevelopmentalEncoding,
    targets: list[nx.DiGraph[int]],
) -> Population:
    """Assign distance-based fitness to individuals requiring evaluation."""
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
    """Apply mutation; k=0 is the baseline."""
    mutation_probability = 0.0 if k == 0 else MUTATION_PROBABILITY
    for individual in offspring:
        if rng.random() < mutation_probability:
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


def get_run_directory(seed: int, k: int) -> Path:
    """Create the output directory for one seed and mutation condition."""
    directory = RESULTS_DIR / f"k_{k}" / f"seed_{seed}"
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
        "condition_mutation_probability": 0.0 if k == 0 else MUTATION_PROBABILITY,
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


def load_histories_for_k(k: int) -> list[list[dict[str, Any]]]:
    """Load all seed histories for one mutation condition."""
    return [
        json.loads(
            (RESULTS_DIR / f"k_{k}" / f"seed_{seed}" / "history.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in SEEDS
    ]


def plot_k_results(k: int) -> None:
    """Plot mean and best fitness over generations for one k value."""
    import matplotlib.pyplot as plt

    histories = load_histories_for_k(k)
    # Exclude generation zero so the plot has exactly NUM_GENERATIONS points.
    rows = [history[1:] for history in histories]
    mean_values = np.asarray([[row["mean"] for row in history] for history in rows])
    best_values = np.asarray([[row["best"] for row in history] for history in rows])
    generations = np.arange(1, NUM_GENERATIONS + 1)

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for axis, values, title in (
        (axes[0], mean_values, "Average population fitness"),
        (axes[1], best_values, "Best population fitness"),
    ):
        average = values.mean(axis=0)
        spread = values.std(axis=0)
        axis.plot(generations, average)
        axis.fill_between(generations, average - spread, average + spread, alpha=0.2)
        axis.set_ylabel("Fitness")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[1].set_xlabel("Generation")
    figure.suptitle(f"Fitness over generations (k={k})")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / f"k_{k}" / "fitness_by_generation.png", dpi=150)
    plt.close(figure)


def plot_all_k_results() -> None:
    """Create aggregate plots and the final-fitness table across k values."""
    import matplotlib.pyplot as plt

    mean_by_k: dict[int, np.ndarray] = {}
    best_by_k: dict[int, np.ndarray] = {}
    final_rows: list[dict[str, float | int]] = []
    generations = np.arange(1, NUM_GENERATIONS + 1)

    for k in K:
        histories = load_histories_for_k(k)
        rows = [history[1:] for history in histories]
        mean_values = np.asarray([[row["mean"] for row in history] for history in rows])
        best_values = np.asarray([[row["best"] for row in history] for history in rows])
        mean_by_k[k] = mean_values.mean(axis=0)
        best_by_k[k] = best_values.mean(axis=0)
        final_rows.append(
            {
                "k": k,
                "average_final_fitness": float(mean_by_k[k][-1]),
                "best_final_fitness": float(best_by_k[k][-1]),
                "average_final_fitness_std": float(mean_values[:, -1].std()),
                "best_final_fitness_std": float(best_values[:, -1].std()),
            }
        )

    for name, values, ylabel in (
        ("average_fitness_all_k.png", mean_by_k, "Average population fitness"),
        ("best_fitness_all_k.png", best_by_k, "Best population fitness"),
    ):
        figure, axis = plt.subplots(figsize=(10, 6))
        for k, series in values.items():
            axis.plot(generations, series, label=f"k={k}")
        axis.set(xlabel="Generation", ylabel="Fitness", title=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(RESULTS_DIR / name, dpi=150)
        plt.close(figure)

    x = np.arange(len(K))
    width = 0.38
    figure, axis = plt.subplots(figsize=(10, 6))
    average_final = [row["average_final_fitness"] for row in final_rows]
    best_final = [row["best_final_fitness"] for row in final_rows]
    axis.bar(x - width / 2, average_final, width, label="Average fitness")
    axis.bar(x + width / 2, best_final, width, label="Best fitness")
    axis.set_xticks(x, [str(k) for k in K])
    axis.set(
        xlabel="k",
        ylabel="Final-generation fitness",
        title="Final fitness by condition",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "final_fitness_by_k.png", dpi=150)
    plt.close(figure)

    with (RESULTS_DIR / "final_fitness_by_k.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    (RESULTS_DIR / "final_fitness_by_k.json").write_text(
        json.dumps(final_rows, indent=2), encoding="utf-8"
    )


"""RUN THE ACTUAL ALGORITHM"""


def main() -> None:
    targets = get_targets()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    """
    for seed in SEEDS:
        nde = create_nde(seed)
        for k in K:
            rng = seed_operators(seed)
            population = initialise_pop(seed)
            evaluate_population(population, nde, targets)
            history: list[dict[str, Any]] = []
            record_generation(population, seed, k, history)

            for _ in range(NUM_GENERATIONS):
                population = produce_offspring(population, k, rng)
                evaluate_population(population, nde, targets)
                population = murder_majority(population)
                record_generation(population, seed, k, history)
            save_run_results(population, nde, targets, seed, k, history)

    """
    for k in K:
        plot_k_results(k)
    plot_all_k_results()


if __name__ == "__main__":
    main()
