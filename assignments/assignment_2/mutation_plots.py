from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np


DEFAULT_PROBABILITIES = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_SEEDS = (42, 7, 1124, 8486, 2026)
YScale = Literal["linear", "symlog"]

PLOT_STYLE = {
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


@dataclass(frozen=True)
class MutationHistory:
    probability: float
    seeds: tuple[int, ...]
    generations: np.ndarray
    mean: np.ndarray  # shape: (number of seeds, number of generations)
    best: np.ndarray  # each row contains the population MINIMUM for one seed


def _read_history(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one run and reject incomplete or malformed generation sequences."""
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"generation", "mean", "best"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required CSV columns are {sorted(required)}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: no generations found")

    try:
        generations = np.asarray([int(row["generation"]) for row in rows])
        mean = np.asarray([float(row["mean"]) for row in rows], dtype=float)
        best = np.asarray([float(row["best"]) for row in rows], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid generation or fitness value") from error

    # The attached recorder writes exactly 0, 1, ..., GENERATIONS.
    if not np.array_equal(generations, np.arange(len(rows))):
        raise ValueError(f"{path}: expected consecutive generations starting at 0")
    if generations[-1] < 1:
        raise ValueError(f"{path}: only the initial population is present")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(best)):
        raise ValueError(f"{path}: NaN/Inf found; check the run before comparing it")
    if np.any(mean < 0) or np.any(best < 0):
        raise ValueError(f"{path}: negative values do not match this fitness function")
    if np.any(best > mean + 1e-10 * np.maximum(1.0, mean)):
        raise ValueError(f"{path}: best must be a minimum because lower is better")

    # Large INVALID_FITNESS penalties are deliberately retained.
    return generations, mean, best


def load_histories_for_mutation(
    experiment_dir: Path,
    probability: float,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    *,
    expected_generations: int | None = None,
) -> MutationHistory:
    """Load matching CSV histories for one per-weight mutation probability."""
    experiment_dir = Path(experiment_dir)
    probability = float(probability)
    seed_values = tuple(seeds)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Mutation probabilities must be finite and between 0 and 1")
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("Provide at least one seed, without duplicate seeds")

    loaded = []
    generations = None
    for seed in seed_values:
        path = (
            experiment_dir
            / f"mutation_{probability}"
            / f"seed_{seed}"
            / "fitness_data"
            / "fitness.csv"
        )
        current, mean, best = _read_history(path)
        if generations is None:
            generations = current
        elif not np.array_equal(generations, current):
            raise ValueError(f"{path}: generations differ between seeds")
        if expected_generations is not None and current[-1] != expected_generations:
            raise ValueError(
                f"{path}: ends at generation {current[-1]}, "
                f"expected {expected_generations}"
            )
        loaded.append((mean, best))

    assert generations is not None
    return MutationHistory(
        probability=probability,
        seeds=seed_values,
        generations=generations,
        mean=np.stack([row[0] for row in loaded]),
        best=np.stack([row[1] for row in loaded]),
    )


def _mean_and_sd(values: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """SD is undefined for one seed; never invent a zero in that case."""
    average = values.mean(axis=0)
    spread = values.std(axis=0, ddof=1) if len(values) > 1 else None
    return average, spread


def _style_axis(axis, yscale: YScale, *, generations: bool = False) -> None:
    from matplotlib.ticker import MaxNLocator

    axis.set_ylabel("Fitness (kleiner = besser)")
    if yscale == "symlog":
        # Linear near zero, logarithmic at large magnitudes; zero remains valid.
        axis.set_yscale("symlog", linthresh=1.0)
        axis.set_ylabel("Fitness (kleiner = besser; symlog)")
    axis.grid(axis="both", alpha=0.25)
    axis.set_axisbelow(True)
    if generations:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    # No fixed y limits or fitness clipping: all seed results stay visible.


def _save_figure(figure, output_dir: Path, name: str) -> None:
    figure.savefig(output_dir / f"{name}.png", dpi=300)
    figure.savefig(output_dir / f"{name}.pdf")


def plot_mutation_results(
    history: MutationHistory,
    output_dir: Path,
    *,
    include_generation_zero: bool = True,
    yscale: YScale = "linear",
    color: str = "C0",
) -> None:
    """Plot individual seeds and their mean +/- SD for one condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if yscale not in ("linear", "symlog"):
        raise ValueError("yscale must be 'linear' or 'symlog'")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = history.generations >= (0 if include_generation_zero else 1)
    generations = history.generations[mask]

    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        for axis, values, title in (
            (axes[0], history.mean[:, mask], "Mittlere Populationsfitness"),
            (axes[1], history.best[:, mask], "Beste Populationsfitness (Minimum)"),
        ):
            for index, series in enumerate(values):
                axis.plot(
                    generations, series, color="0.55", linewidth=1, alpha=0.5,
                    label="Einzelne Seeds" if index == 0 else None,
                )
            average, spread = _mean_and_sd(values)
            axis.plot(
                generations, average, color=color, linewidth=2.5,
                label="Mittelwert über Seeds",
            )
            if spread is not None:
                axis.fill_between(
                    generations, average - spread, average + spread,
                    color=color, alpha=0.2, label="±1 SD zwischen Seeds",
                )
            axis.set_title(title)
            _style_axis(axis, yscale, generations=True)
            axis.legend(loc="best")

        xlabel = "Generation (0 = Startpopulation)" if include_generation_zero else "Generation"
        axes[1].set_xlabel(xlabel)
        figure.suptitle(
            f"Mutationswahrscheinlichkeit p={history.probability:g} "
            f"({history.probability:.0%}); n={len(history.seeds)} Seeds"
        )
        figure.tight_layout()
        token = str(history.probability).replace(".", "_")
        _save_figure(figure, output_dir, f"fitness_mutation_{token}")
        plt.close(figure)


def plot_all_mutation_results(
    experiment_dir: Path,
    values: Iterable[float] = DEFAULT_PROBABILITIES,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    *,
    include_generation_zero: bool = True,
    yscale: YScale = "linear",
    expected_generations: int | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Create condition plots, comparison plots, and auditable final tables.

    All runs must have the same generation sequence. Nothing is silently
    skipped, shortened, or filtered. With one seed no SD is drawn, and the
    summary tables use null/empty for undefined SDs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if yscale not in ("linear", "symlog"):
        raise ValueError("yscale must be 'linear' or 'symlog'")
    experiment_dir = Path(experiment_dir)
    probabilities = tuple(sorted(float(value) for value in values))
    seed_values = tuple(seeds)
    if not probabilities or len(set(probabilities)) != len(probabilities):
        raise ValueError("Provide at least one mutation probability, without duplicates")

    # Validate every run BEFORE creating plots or summary tables.
    histories = [
        load_histories_for_mutation(
            experiment_dir, probability, seed_values,
            expected_generations=expected_generations,
        )
        for probability in probabilities
    ]
    reference = histories[0].generations
    for history in histories[1:]:
        if not np.array_equal(reference, history.generations):
            raise ValueError(
                f"p={history.probability}: generations differ between conditions"
            )

    output_dir = Path(output_dir) if output_dir is not None else experiment_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = [f"C{index % 10}" for index in range(len(histories))]
    mask = reference >= (0 if include_generation_zero else 1)
    generations = reference[mask]
    n_seeds = len(seed_values)
    statistics_label = "Mittelwert ± 1 SD" if n_seeds > 1 else "Ein Seed; keine SD"
    xlabel = "Generation (0 = Startpopulation)" if include_generation_zero else "Generation"

    for history, color in zip(histories, colors):
        plot_mutation_results(
            history, output_dir,
            include_generation_zero=include_generation_zero,
            yscale=yscale, color=color,
        )

    with plt.rc_context(PLOT_STYLE):
        # Mean-of-population and best-of-population have separate figures.
        for name, metric, title in (
            ("average_fitness_all_mutations", "mean", "Mittlere Populationsfitness"),
            ("best_fitness_all_mutations", "best", "Beste Populationsfitness (Minimum)"),
        ):
            figure, axis = plt.subplots(figsize=(12, 7))
            for history, color in zip(histories, colors):
                average, spread = _mean_and_sd(getattr(history, metric)[:, mask])
                axis.plot(
                    generations, average, color=color, linewidth=2.3,
                    linestyle="--" if history.probability == 0 else "-",
                    label=f"p={history.probability:g} ({history.probability:.0%})",
                )
                if spread is not None:
                    axis.fill_between(
                        generations, average - spread, average + spread,
                        color=color, alpha=0.12,
                    )
            axis.set_title(f"{title}\n{statistics_label} über n={n_seeds} Seeds")
            axis.set_xlabel(xlabel)
            _style_axis(axis, yscale, generations=True)
            axis.legend(title="Mutation pro Gewicht", ncol=min(3, len(histories)))
            figure.tight_layout()
            _save_figure(figure, output_dir, name)
            plt.close(figure)

        # Dot plots expose the five runs; bars would hide their distribution.
        x = np.arange(len(histories))
        offsets = np.linspace(-0.14, 0.14, n_seeds) if n_seeds > 1 else np.zeros(1)
        figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        for axis, metric, title in (
            (axes[0], "mean", "Mittlere Populationsfitness am Ende"),
            (axes[1], "best", "Beste Populationsfitness am Ende"),
        ):
            final_values = np.stack([getattr(history, metric)[:, -1] for history in histories])
            for index, color in enumerate(colors):
                axis.scatter(
                    index + offsets, final_values[index], color=color,
                    s=55, alpha=0.8, zorder=3,
                    label="Einzelne Seeds" if index == 0 else None,
                )
            average, spread = _mean_and_sd(final_values.T)
            axis.errorbar(
                x, average, yerr=spread, fmt="D", color="black", markersize=7,
                capsize=6, linewidth=1.8, zorder=4, label=statistics_label,
            )
            axis.set_title(title)
            axis.set_xticks(x, [f"{probability:.0%}" for probability in probabilities])
            axis.set_xlim(-0.5, len(histories) - 0.5)
            _style_axis(axis, yscale)
            axis.legend(loc="best")
        axes[1].set_xlabel("Mutationswahrscheinlichkeit pro Gewicht")
        figure.suptitle(f"Endvergleich: Generation {reference[-1]}, n={n_seeds} Seeds")
        figure.tight_layout()
        _save_figure(figure, output_dir, "final_fitness_by_mutation")
        plt.close(figure)

    summary_rows = []
    seed_rows = []
    for history in histories:
        final_mean = history.mean[:, -1]
        final_best = history.best[:, -1]
        summary_rows.append({
            "mutation_probability": history.probability,
            "n_seeds": n_seeds,
            "final_generation": int(reference[-1]),
            "average_final_fitness": float(final_mean.mean()),
            "average_final_fitness_sd_between_seeds": (
                float(final_mean.std(ddof=1)) if n_seeds > 1 else None
            ),
            "best_final_fitness_mean": float(final_best.mean()),
            "best_final_fitness_sd_between_seeds": (
                float(final_best.std(ddof=1)) if n_seeds > 1 else None
            ),
            "best_final_fitness_median": float(np.median(final_best)),
            "best_final_fitness_min_across_seeds": float(final_best.min()),
        })
        for index, seed in enumerate(seed_values):
            seed_rows.append({
                "mutation_probability": history.probability,
                "seed": seed,
                "final_generation": int(reference[-1]),
                "mean_final_fitness": float(final_mean[index]),
                "best_final_fitness": float(final_best[index]),
            })

    for name, rows in (
        ("final_fitness_by_mutation.csv", summary_rows),
        ("final_fitness_per_seed.csv", seed_rows),
    ):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "final_fitness_by_mutation.json").write_text(
        json.dumps(summary_rows, indent=2, allow_nan=False), encoding="utf-8"
    )
    return output_dir


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--values", type=float, nargs="+", default=DEFAULT_PROBABILITIES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--exclude-generation-zero", action="store_true")
    parser.add_argument("--yscale", choices=("linear", "symlog"), default="linear")
    parser.add_argument("--expected-generations", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        output_dir = plot_all_mutation_results(
            args.experiment_dir, args.values, args.seeds,
            include_generation_zero=not args.exclude_generation_zero,
            yscale=args.yscale,
            expected_generations=args.expected_generations,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Plots und Tabellen erstellt: {output_dir.resolve()}")


if __name__ == "__main__":
    _main()
