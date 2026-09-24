"""Compare five crossover conditions for a fixed John Set gecko."""

import argparse
import csv
import json
import random
from pathlib import Path

import mujoco as mj
import numpy as np

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import gecko
from ariel.ec import (
    EA, Crossover, EAOperation, FloatMutator, FloatsGenerator,
    Individual, Population, set_seed,
)
from ariel.simulation.environments import OlympicArena
from ariel.simulation.tasks.targeted_locomotion import distance_to_target

CONDITIONS = (
    "none", "local_discrete", "local_intermediate",
    "global_discrete", "global_intermediate",
)
SPAWN = [-0.8, 0.0, 0.1]
TARGET = np.array([5.42, 0.0, 0.0])


def make_individual(weights) -> Individual:
    """Copy a weight vector into a new, unevaluated candidate."""
    individual = Individual()
    individual.genotype = list(weights)
    return individual


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


class Evaluator:
    """Reuse a compiled arena while resetting simulation state per candidate."""

    def __init__(self, hidden: int, duration: float):
        """Build the fixed body/world and determine network and simulation sizes."""
        mj.set_mjcb_control(None)
        world = OlympicArena()
        world.spawn(gecko().spec, position=SPAWN, correct_collision_with_floor=True)
        self.model = world.spec.compile()
        self.hidden = hidden
        self.steps = int(np.ceil(duration / self.model.opt.timestep))
        self.inputs = self.model.nq + 4
        self.split = self.inputs * hidden
        self.size = self.split + hidden * self.model.nu

    def evaluate(self, population: Population) -> Population:
        """Score unevaluated controllers by final planar distance; lower is better."""
        for individual in population.unevaluated:
            # Decode input-to-hidden and hidden-to-output matrices, then reset.
            weights = np.asarray(individual.genotype)
            w1 = weights[:self.split].reshape(self.inputs, self.hidden)
            w2 = weights[self.split:].reshape(self.hidden, self.model.nu)
            data = mj.MjData(self.model)
            mj.mj_resetData(self.model, data)
            mj.mj_forward(self.model, data)

            def control(model, state):
                """Map joint/body positions, target offset, and a 1 Hz clock to hinge angles."""
                phase = 2 * np.pi * state.time
                inputs = np.concatenate((
                    state.qpos, TARGET[:2] - state.qpos[:2],
                    [np.sin(phase), np.cos(phase)],
                ))
                state.ctrl[:] = np.tanh(np.tanh(inputs @ w1) @ w2) * (np.pi / 2)

            try:
                # Run the fixed duration and always release MuJoCo's global callback.
                mj.set_mjcb_control(control)
                mj.mj_step(self.model, data, nstep=self.steps)
            finally:
                mj.set_mjcb_control(None)
            if (not np.isfinite(data.qpos).all()
                    or data.time < (self.steps - 0.5) * self.model.opt.timestep):
                raise FloatingPointError("Simulation diverged; run stopped.")
            individual.fitness = distance_to_target(data.qpos[:3], TARGET)
        return population


def select_survivors(population: Population, size: int) -> Population:
    """Keep the best parents/offspring alive; retain losers for database history."""
    survivors = {id(ind) for ind in population.best(sort="min", n=size)}
    for individual in population:
        individual.alive = id(individual) in survivors
    return population


def run(args, evaluator: Evaluator, condition: str, seed: int) -> None:
    """Run one condition/seed with a fixed budget and save settings and results."""
    # Use a separate directory per run; refuse to overwrite previous results.
    output = args.output / condition / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    settings = vars(args) | {
        "output": str(output), "condition": condition, "seed": seed,
        "body": "john_set.gecko", "spawn": SPAWN, "target": TARGET.tolist(),
        "genotype_size": evaluator.size, "timestep": evaluator.model.opt.timestep,
        "simulation_steps": evaluator.steps,
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2))
    # Start every condition with the same evaluated population for this seed.
    set_seed(seed)
    random.seed(seed)
    population = Population([
        make_individual(FloatsGenerator.normal(std=0.5, size=evaluator.size))
        for _ in range(args.population)
    ])
    evaluator.evaluate(population)
    generation = 0

    def reproduce(population: Population) -> Population:
        """Create the offspring batch, mutate it, and combine it with the parents."""
        children = Population([
            make_individual(recombine(population, condition))
            for _ in range(args.offspring)
        ])
        # Match mutation draws across conditions independently of crossover draws.
        set_seed(int(np.random.SeedSequence([seed, generation, 1]).generate_state(1)[0]))
        for child in children:
            child.genotype = FloatMutator.gaussian(
                child.genotype, std=args.mutation_std,
                mutation_probability=args.mutation_probability,
            )
        return population + children

    # Each generation: select/recombine/mutate, evaluate, then retain survivors.
    operations = [
        EAOperation(reproduce), EAOperation(evaluator.evaluate),
        EAOperation(select_survivors, args.population),
    ]
    ea = EA(population, operations, is_maximisation=False, quiet=True,
            db_file_path=output / "population.db", db_handling="halt")
    try:
        with (output / "history.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["generation", "evaluations", "best", "mean", "std", "worst"])
            # Include generation zero; statistics describe the surviving population.
            for generation in range(args.generations + 1):
                if generation:
                    ea.step()
                ea.fetch_population()
                fitness = np.array([ind.fitness for ind in ea.population.alive])
                writer.writerow([
                    generation, args.population + generation * args.offspring,
                    fitness.min(), fitness.mean(), fitness.std(), fitness.max(),
                ])
                stream.flush()
                print(f"{condition} seed={seed} generation={generation} best={fitness.min():.4f}", flush=True)
        # Export the winning controller separately for later replay.
        best = ea.population.alive.best(sort="min")[0]
        (output / "best.json").write_text(json.dumps({
            "fitness": best.fitness, "weights": best.genotype,
        }))
    finally:
        ea.engine.dispose()


def plot_results(output: Path) -> None:
    """TODO: plot mean and standard deviation across independent seed runs."""
    raise NotImplementedError


def main() -> None:
    """Validate CLI settings and run every requested condition for each seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--offspring", type=int, default=20)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--hidden", type=int, default=6)
    parser.add_argument("--mutation-std", type=float, default=0.1)
    parser.add_argument("--mutation-probability", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if (args.population < 2 or args.offspring < 1 or args.generations < 0
            or args.hidden < 1 or not np.isfinite(args.duration) or args.duration <= 0
            or not np.isfinite(args.mutation_std) or args.mutation_std < 0
            or not 0 <= args.mutation_probability <= 1 or min(args.seeds) < 0):
        parser.error("Invalid population, budget, controller, mutation, or seed settings.")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.conditions)) != len(args.conditions):
        parser.error("Seeds and conditions must be unique.")
    evaluator = Evaluator(args.hidden, args.duration)
    for seed in args.seeds:
        for condition in args.conditions:
            run(args, evaluator, condition, seed)


if __name__ == "__main__":
    main()
