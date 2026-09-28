"""Compare five crossover conditions for a fixed John Set gecko."""

import csv
import json
import random
import sys
from pathlib import Path

# Allow direct execution from the assignment directory.
ARIEL_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(ARIEL_SRC))

import mujoco as mj
import numpy as np

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import gecko
from ariel.ec import (
    EA, Crossover, EAOperation, FloatMutator, FloatsGenerator,
    Individual, Population, set_seed,
)
from ariel.simulation.environments import OlympicArena
from ariel.simulation.tasks.targeted_locomotion import distance_to_target

# Settings and constants
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
POPULATION_SIZE = 20
OFFSPRING_SIZE = 20
NUM_GENERATIONS = 100
SEEDS = list(range(5))
INIT_SD = 0.5
MUTATION_PROBABILITY = 0.1
MUTATION_SD = 0.1
HIDDEN_SIZE = 6
SIM_DURATION = 15.0
PLOT_ONLY = False
PLOT_STYLE = {"font.size": 14, "legend.fontsize": 11}

CONDITIONS = (
    "none", "local_discrete", "local_intermediate",
    "global_discrete", "global_intermediate",
)
SPAWN = [-0.8, 0.0, 0.1]
TARGET = np.array([5.42, 0.0, 0.0])


# Evolutionary steps

def seed_operators(seed: int) -> None:
    """Reset Python and Ariel randomness for one independent run."""
    random.seed(seed)
    set_seed(seed)


def make_individual(weights) -> Individual:
    """Copy weights into a fresh, unevaluated candidate."""
    individual = Individual()
    individual.genotype = list(weights)
    return individual


def build_model() -> mj.MjModel:
    """Compile the fixed John Set gecko in OlympicArena once."""
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(gecko().spec, position=SPAWN, correct_collision_with_floor=True)
    return world.spec.compile()


def initialise_pop(model: mj.MjModel) -> Population:
    """Sample one normally distributed weight vector per controller."""
    size = (model.nq + 4 + model.nu) * HIDDEN_SIZE
    return Population([make_individual(FloatsGenerator.normal(std=INIT_SD, size=size))
                       for _ in range(POPULATION_SIZE)])


def decode_genome(genotype, model: mj.MjModel):
    """Reshape the flat genotype into the two neural-network weight matrices."""
    weights = np.asarray(genotype)
    split = (model.nq + 4) * HIDDEN_SIZE
    return weights[:split].reshape(-1, HIDDEN_SIZE), weights[split:].reshape(HIDDEN_SIZE, model.nu)


def nn_controller(state, weights):
    """Map positions, target offset, and a 1 Hz clock to hinge angles."""
    phase = 2 * np.pi * state.time
    inputs = np.concatenate((state.qpos, TARGET[:2] - state.qpos[:2],
                             [np.sin(phase), np.cos(phase)]))
    w1, w2 = weights
    return np.tanh(np.tanh(inputs @ w1) @ w2) * (np.pi / 2)


def evaluate_population(population: Population, model: mj.MjModel) -> Population:
    """Reset and simulate each unevaluated controller; score final planar distance."""
    steps = int(np.ceil(SIM_DURATION / model.opt.timestep))
    for individual in population.unevaluated:
        weights = decode_genome(individual.genotype, model)
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

        def control(model, state):
            """Apply the current controller at each physics step."""
            state.ctrl[:] = nn_controller(state, weights)

        try:
            mj.set_mjcb_control(control)
            mj.mj_step(model, data, nstep=steps)
        finally:
            mj.set_mjcb_control(None)
        if not np.isfinite(data.qpos).all() or data.time < (steps - 0.5) * model.opt.timestep:
            raise FloatingPointError("Simulation diverged; run stopped.")
        individual.fitness = distance_to_target(data.qpos[:3], TARGET)
    return population


def recombine(parents: Population, condition: str) -> list[float]:
    """Select parents uniformly and produce one offspring weight vector."""
    if condition == "none":
        return list(parents.sample(1)[0].genotype)
    if condition.startswith("local_"):
        # Local: one distinct parent pair for the entire vector.
        a, b = parents.sample(2)
        if condition == "local_discrete":
            return Crossover.uniform(a.genotype, b.genotype)[0]
        if condition == "local_intermediate":
            return ((np.array(a.genotype) + b.genotype) / 2).tolist()
    if condition not in ("global_discrete", "global_intermediate"):
        raise ValueError(f"Unknown condition: {condition}")
    weights = []
    # Global: a fresh distinct pair per weight; copy or average its values.
    for i in range(len(parents[0].genotype)):
        a, b = parents.sample(2)
        weights.append(
            random.choice((a.genotype[i], b.genotype[i]))
            if condition == "global_discrete"
            else (a.genotype[i] + b.genotype[i]) / 2
        )
    return weights


def create_offspring(population: Population, condition: str) -> Population:
    """Select parents and create the fixed offspring batch for this condition."""
    return Population([make_individual(recombine(population, condition))
                       for _ in range(OFFSPRING_SIZE)])


def mutate_population(offspring: Population, seed: int, generation: int) -> Population:
    """Apply Gaussian mutation with matching draws across crossover conditions."""
    set_seed(int(np.random.SeedSequence([seed, generation, 1]).generate_state(1)[0]))
    for child in offspring:
        child.genotype = FloatMutator.gaussian(
            child.genotype, std=MUTATION_SD, mutation_probability=MUTATION_PROBABILITY)
        child.requires_eval = True
    return offspring


def produce_offspring(population: Population, condition: str, seed: int, history: list) -> Population:
    """Append mutated children while retaining parents for survivor selection."""
    offspring = create_offspring(population, condition)
    return population + mutate_population(offspring, seed, len(history))


def select_survivors(population: Population) -> Population:
    """Retain the best living parents/offspring and archive the rest as dead."""
    for individual in population.alive.sort(sort="min")[POPULATION_SIZE:]:
        individual.alive = False
    return population


# Recording and output

def record_generation(population: Population, history: list) -> Population:
    """Append survivor statistics, starting with generation zero."""
    fitness = np.array([ind.fitness for ind in population.alive])
    history.append(dict(generation=len(history), evaluations=POPULATION_SIZE + len(history) * OFFSPRING_SIZE,
                        best=float(fitness.min()), mean=float(fitness.mean()),
                        std=float(fitness.std()), worst=float(fitness.max())))
    return population


def get_run_directory(seed: int, condition: str) -> Path:
    """Return the output location for one seed and crossover condition."""
    return RESULTS_DIR / condition / f"seed_{seed}"


def save_run_results(population: Population, model: mj.MjModel, seed: int, condition: str, history: list) -> None:
    """Save run settings, generation statistics, and the best controller."""
    directory = get_run_directory(seed, condition)
    best = population.alive.best(sort="min")[0]
    settings = dict(seed=seed, condition=condition, population=POPULATION_SIZE, offspring=OFFSPRING_SIZE,
                    generations=NUM_GENERATIONS, hidden=HIDDEN_SIZE, duration=SIM_DURATION,
                    mutation_std=MUTATION_SD, mutation_probability=MUTATION_PROBABILITY,
                    init_std=INIT_SD, body="john_set.gecko", spawn=SPAWN, target=TARGET.tolist(),
                    genotype_size=len(best.genotype), timestep=model.opt.timestep,
                    simulation_steps=int(np.ceil(SIM_DURATION / model.opt.timestep)))
    for name, payload in (("settings", settings), ("history", history),
                          ("best", dict(fitness=best.fitness, weights=best.genotype))):
        (directory / f"{name}.json").write_text(json.dumps(payload, indent=2))
    with (directory / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


# Plotting

def load_histories_for_condition(condition: str):
    """Load all configured seed histories for one crossover condition."""
    return [np.genfromtxt(get_run_directory(seed, condition) / "history.csv",
                          delimiter=",", names=True, ndmin=1) for seed in SEEDS]


def plot_results() -> None:
    """Plot matching seed histories; runs must share the same generation budget."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output, conditions, seeds = RESULTS_DIR, CONDITIONS, SEEDS
    results, final_rows = {}, []
    for condition in conditions:
        histories = load_histories_for_condition(condition)
        values = np.array([np.column_stack((h["mean"], h["best"])) for h in histories])
        results[condition] = (values.mean(axis=0), values.std(axis=0))
        average, spread = results[condition]
        final_rows.append({
            "condition": condition, "seeds": len(histories),
            "average_final_fitness": float(average[-1, 0]),
            "best_final_fitness": float(average[-1, 1]),
            "average_final_fitness_std": float(spread[-1, 0]),
            "best_final_fitness_std": float(spread[-1, 1]),
        })

    # Include generation zero to show the common initial population.
    generations = histories[0]["generation"]
    labels = {c: "No crossover" if c == "none" else c.replace("_", " ").capitalize()
              for c in conditions}
    titles = ("Average population fitness", "Best population fitness")
    ylabel = "Distance to target (m)\n(lower is better)"
    note = f"Mean ± SD across {len(seeds)} seeds"

    def save(figure, path):
        """Lay out, save, and close a figure."""
        figure.tight_layout()
        figure.savefig(output / path, dpi=150)
        plt.close(figure)

    with plt.rc_context(PLOT_STYLE):
        for condition, (average, spread) in results.items():
            figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            for i, axis in enumerate(axes):
                axis.plot(generations, average[:, i])
                axis.fill_between(generations, average[:, i] - spread[:, i],
                                  average[:, i] + spread[:, i], alpha=0.2)
                axis.set(ylabel=ylabel, title=titles[i])
                axis.grid(alpha=0.25)
            axes[-1].set_xlabel("Generation")
            figure.suptitle(f"{labels[condition]}\n{note}")
            save(figure, Path(condition) / "fitness_by_generation.png")

        for i, metric in enumerate(("average", "best")):
            figure, axis = plt.subplots(figsize=(11, 7))
            for condition, (average, spread) in results.items():
                line, = axis.plot(generations, average[:, i], label=labels[condition])
                axis.fill_between(generations, average[:, i] - spread[:, i],
                                  average[:, i] + spread[:, i], color=line.get_color(), alpha=0.12)
            axis.set(xlabel="Generation", ylabel=ylabel, title=f"{titles[i]}\n{note}")
            axis.grid(alpha=0.25)
            axis.legend()
            save(figure, f"{metric}_fitness_all_conditions.png")

        figure, axis = plt.subplots(figsize=(12, 7))
        x, width = np.arange(len(conditions)), 0.38
        for i, metric in enumerate(("average", "best")):
            axis.bar(x + (i - 0.5) * width,
                     [row[f"{metric}_final_fitness"] for row in final_rows], width,
                     yerr=[row[f"{metric}_final_fitness_std"] for row in final_rows],
                     capsize=4, label=titles[i])
        axis.set_xticks(x, [labels[c].replace(" ", "\n", 1) for c in conditions])
        axis.set(ylabel=ylabel, title=f"Final fitness by crossover condition\n{note}")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend()
        save(figure, "final_fitness_by_condition.png")

    with (output / "final_fitness_by_condition.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    (output / "final_fitness_by_condition.json").write_text(json.dumps(final_rows, indent=2))


# Run the experiment

def main() -> None:
    """Run every seed/condition, then generate the comparison graphs."""
    if not PLOT_ONLY:
        model = build_model()
        for seed in SEEDS:
            for condition in CONDITIONS:
                directory = get_run_directory(seed, condition)
                directory.mkdir(parents=True, exist_ok=False)
                seed_operators(seed)
                population = initialise_pop(model)
                evaluate_population(population, model)
                history = []
                record_generation(population, history)
                operations = [EAOperation(produce_offspring, condition, seed, history),
                              EAOperation(evaluate_population, model), EAOperation(select_survivors),
                              EAOperation(record_generation, history)]
                ea = EA(population, operations, is_maximisation=False, quiet=True,
                        db_file_path=directory / "population.db", db_handling="halt")
                try:
                    for generation in range(1, NUM_GENERATIONS + 1):
                        ea.step()
                        print(f"{condition} seed={seed} generation={generation} best={history[-1]['best']:.4f}", flush=True)
                    ea.fetch_population()
                    save_run_results(ea.population, model, seed, condition, history)
                finally:
                    ea.engine.dispose()
    plot_results()


if __name__ == "__main__":
    main()
