# Assignment 2: snake brain evolution

Use the existing project `.venv` (Python 3.12+). From `evolutionary_computing`:

```powershell
uv run python assignments/assignment_2/A2_template_2026.py
```

This trains a neural-network controller and then opens a looping MuJoCo replay
of the best saved brain. Close the viewer to stop. The red pole/yellow ball is
the target. The John Set snake body stays fixed; only its brain evolves.

Replay without training again:

```powershell
uv run python assignments/assignment_2/A2_template_2026.py --replay
```

Add `--video` to export an MP4 and preview image. Add `--no-view` for headless
training or verification. `--replay PATH_TO_BRAIN_JSON` selects an older run.

## Generic starting configuration

- Body: unchanged `john_set.snake`, eight hinges, tail behind the core.
- World: selected in `build_world()`, currently SimpleFlatWorld following your
  map change. The original OlympicArena option remains available via
  `_FixedOlympicArena(load_precompiled=False)`. ARIEL source is unchanged.
- Spawn: core X/Y `(0, 0)`; target X/Y `(2, 0)`.
- Evaluation: 15 simulated seconds at the stock 0.002-second physics timestep.
- Fitness: final core-to-target XY distance in metres, minimised. It does not
  measure the closest approach, whole-body distance, or completion of the stadium.
- Brain: 36 inputs, 8 tanh hidden units, 8 tanh outputs scaled to hinge limits;
  360 evolved weights including biases. Inputs contain orientation, hinge angles,
  velocities, target displacement, sine/cosine clocks at two frequencies,
  elapsed time divided by evaluation duration, and sine/cosine of the signed
  heading error to the target. Heading uses core-local -X (away from the tail);
  elapsed time is bounded to [0, 1]. Training and replay use the same duration.
  These features give evolution direct timing and steering information. There is no
  CPG or preprogrammed gait. Direct angle control runs every physics step.
- EA: population 32, tournament size 3, uniform crossover probability 0.2,
  independent per-weight mutation probability 0.1, Gaussian sigma 0.15,
  weights clipped to `[-5, 5]`, elitist `(mu + lambda)` survival.
- Stop: 15 generations without at least 0.0001 m improvement, or a maximum of
  60 offspring generations. The saved result says which limit stopped the run.
- Budget: 32 initial evaluations plus 32 per offspring generation (maximum 1952).
  Saved-brain verification and replay are separate from the search budget.

Settings are exposed by `--help`; no research question is assumed. A single run
demonstrates a working pipeline, not reliable superiority or general navigation.
The controller is trained for one fixed target and world.

## Continue a trained population

The new inputs require a fresh training run: omit `--resume` and `--replay`
for that first run. Schema 1/2 brains with the old 33-input layout are rejected;
their archives remain untouched. The commands below apply to new schema 3 runs.

```powershell
uv run python assignments/assignment_2/A2_template_2026.py --resume --generations 100 --patience 60 --workers 4
```

`--resume` restores the latest completed run. `--generations 100` now means
**100 additional generations**, not a new run from generation zero. To recover
an interrupted run, pass its printed seed directory or `checkpoint.json` path
explicitly with `--resume PATH`; the default latest pointer refers to a completed
run. Every continuation writes a new directory and preserves its source archive.

New `checkpoint.json` files save the full surviving population, NumPy RNG state,
generation/evaluation counters, and patience state after each complete generation.
Keeping the operator settings unchanged reproduces an uninterrupted run. Older
archives without a checkpoint can recover their population from SQLite, but
start a new seeded random sequence; the program explicitly reports this difference.

Population size, evaluation duration, target, body, and physics must remain
compatible. Continuation loads the archived world even if `build_world()` changed.
Search settings are inherited unless explicitly overridden. A changed search
operator resets patience, but does not discard the existing population. The source
seed remains fixed; continuation does not count as an independent replicate.

Two refinement options are available for comparisons:

```powershell
# Smaller mutations and no crossover, to refine an existing gait.
uv run python assignments/assignment_2/A2_template_2026.py --resume --generations 100 --mutation-sigma 0.03 --crossover-rate 0 --search-mode standard --workers 4

# Mixed mutations: 60% use 0.2x sigma, 30% use 1x, and 10% use 3x.
uv run python assignments/assignment_2/A2_template_2026.py --resume --generations 100 --search-mode mixed --mutation-sigma 0.15 --crossover-rate 0.2 --workers 4
```

Mixed mode disables crossover for the fine-mutation offspring to preserve the
parent's coordinated weights. Standard mode retains the original algorithm.
Neither strategy is guaranteed to outperform the other. Compare from the same
source run and with the same additional evaluation budget; use `--output-dir`
to keep pilot results separate from the main latest-result pointer.

The default success radius is 0.10 m (`--target-radius`): success means final
core XY distance within that radius. It is reported separately and does not
change fitness, stop a trial early, move the target, or extend the 15-second task.

## Later experiments

The assignment requires at least five independent seeds and a baseline. For an
equal-budget comparison, random search uses exactly each EA run's actual number
of evaluations, including when the EA stops on a plateau. Both methods use the
same initial distribution and paired initial populations; random search then
samples entirely fresh chromosomes independently of previous fitness.

```powershell
.\.venv\Scripts\python.exe assignments\assignment_2\A2_template_2026.py --seeds 42 43 44 45 46 --baseline --no-view
```

For a fixed common budget across every seed, add `--patience 0`. Change one
parameter at a time when investigating a research question. The plot shows
mean best-so-far distance and sample standard deviation across seeds; for unequal
length runs it only includes their common observed budgets.

Outputs are under `assignments/assignment_2/__data__/A2_template_2026/`, with a
unique directory for every invocation. Each batch includes the exact binary
model, source snapshot, convergence plot, and summary. Each seed includes:

- `evolution.sqlite`: every evaluated individual and its fitness/lifetime.
- `history.csv`: surviving population best/mean/worst and genotype diversity.
- `config.json`: task, architecture, settings, versions, seeds and checksums.
- `best_brain.json`: reusable chromosome and score, atomically replaced.
- `checkpoint.json`: population, RNG, budget and patience for continuation.
- `trajectory.csv`: winner's time and core X/Y/Z coordinates.
- `result.json`: cumulative/new budgets, elapsed minutes, target success,
  stopping reason, and exact replay verification.

Replaying a saved brain uses the archived physics model and duration and checks
the recorded fitness first. New world labels come from the compiled model.
Legacy records may incorrectly say OlympicArena for a flat-world run; they are
left intact and their archived model checksum identifies the actual world.
Continuation SQLite files contain inherited survivors plus newly evaluated brains;
the original archive retains all earlier candidates. Evaluation counters separately
report inherited work and new work, so continuation is not mistaken for fresh search.

Focused implementation checks:

```powershell
.\.venv\Scripts\python.exe assignments\assignment_2\check_A2_2026.py
```

Pass `--batch PATH_TO_BATCH_DIRECTORY` to also check archive counts and scores.
Pass `--continuation` to compare a split run with an uninterrupted run, including
exact population and RNG equality, and check incompatible-duration rejection.
All original template functions, comments, and docstrings are retained; new
functions start with `_`. The old demo descriptions remain as historical guidance.
