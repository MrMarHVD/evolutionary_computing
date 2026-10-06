# Assignment 2: controlled mutation comparison

Research question: **How do mutation strength and mutation type affect final
fitness, convergence and population diversity when evolving a fixed snake's
neural controller, with the rest of the EA held constant?**

The study uses `A2_template_2026.py` for the actual EA and simulation, and
`mutation_study_A2_2026.py` for the repeated experiment and comparison plots.
It does not run the older multi-parameter tuner.

## Run from PowerShell

From `D:\Masters\EVOLUTIONARY_COMPUTING\evolutionary_computing`:

```powershell
.\.venv\Scripts\python.exe assignments/assignment_2/A2_template_2026.py --mutation-study --workers 4
```

This starts fresh random populations. Defaults are five independent seeds
42-46, 32 parents/32 offspring, and 150 offspring generations: **4,832 search
evaluations per run, 25 runs, 120,800 total search evaluations**. This can take
many hours; the small smoke check below is not an estimate of full-run quality.
The study prints its output directory. It saves videos without opening MuJoCo.

To work in bounded sessions, give the study an explicit new directory:

```powershell
.\.venv\Scripts\python.exe assignments/assignment_2/A2_template_2026.py --mutation-study --workers 4 --max-minutes 120 --output-dir assignments/assignment_2/mutation_studies/main
.\.venv\Scripts\python.exe assignments/assignment_2/A2_template_2026.py --mutation-study --resume assignments/assignment_2/mutation_studies/main --workers 4 --max-minutes 120
```

The time cap pauses search at generation boundaries; it is **not** an unequal
per-condition stopping criterion. Startup, finishing a generation, verification,
plotting and video export can extend wall time. Repeat resume until complete.
`--no-video` defers rendering; a later resume without it exports completed runs
without retraining them. Ctrl+C preserves the last committed checkpoint.
Do not run two processes against the same study directory.

## Default conditions

| Condition | Per-gene probability | Mutation |
|---|---:|---|
| Gaussian, small | 0.10 | add N(0, 0.05^2) |
| Gaussian, medium | 0.10 | add N(0, 0.15^2) |
| Gaussian, large | 0.10 | add N(0, 0.45^2) |
| Uniform reset | 0.10 | replace selected weights with U(-5, 5) |
| Random-search baseline | n/a | independent full brains from clipped N(0, 0.5^2) |

Gaussian strength comparisons change only sigma. Reset versus Gaussian changes
the entire mutation distribution, including locality: do not attribute any
difference solely to distribution shape. Reset has no sigma and is run only once
per rate, regardless of how many sigmas are supplied.

Optional `uniform_step` adds U(-sqrt(3)*sigma, +sqrt(3)*sigma), matching Gaussian
noise variance **before clipping**. It is an additional controlled comparison,
distinct from the textbook's uniform random reset.

Only mutation rates (hold strength and operator fixed):

```powershell
.\.venv\Scripts\python.exe assignments/assignment_2/A2_template_2026.py --mutation-study --operators gaussian --rates 0.01 0.05 0.1 0.3 --sigmas 0.15
```

Only local mutation distributions, matched at sigma 0.15:

```powershell
.\.venv\Scripts\python.exe assignments/assignment_2/A2_template_2026.py --mutation-study --operators gaussian uniform_step --rates 0.1 --sigmas 0.15
```

Combining multiple operators, rates and sigmas forms a factorial grid. The
baseline is added once per seed. Rate zero is a no-mutation EA control, **not**
random search. Empty Bernoulli masks are permitted; no forced edit alters the rate.

## What stays fixed

- John Set snake; seeded OlympicArena (`TERRAIN_SEED=2026`); spawn and final-platform target.
- 360-weight MLP, input features, direct angle control, 15-second episodes,
  0.002-second physics timestep, initial N(0, 0.5^2) weights and [-5,5] bounds.
- Population size, tournament size 3, uniform crossover rate 0.2 with gene
  probability 0.5, and elitist (mu+lambda) survival including duplicate genotypes.
- Simulation duration, fitness, success criterion and evaluation budget.

The same seed gives exactly the same initial population in every condition and
the baseline. Different seeds supply independent replicates. Random streams
diverge after variation, so pairing does not imply identical later random draws.
The same saved model is loaded in every worker.

All study conditions use `search_mode=standard`. Legacy mixed/focused/adaptive
modes are excluded: they also change crossover or survivor behavior and therefore
are unsuitable as a mutation-only comparison. They remain available for older
single-run experiments; the broader tuner and previous results are preserved.

The random baseline samples fresh independent genomes, while retaining its best
samples for reporting. It does not breed from survivors. Verification replays
and videos are additional diagnostic simulations, outside the search budget.

## Assignment and textbook alignment

`Assignment2.pdf`, pp. 2-4, specifies a fixed John Set body/world, an evolved NN
using `ariel.ec`, at least five independent repeats, an equal-budget baseline,
and line plots of mean/std across runs. No `src/ariel` files are changed.

The attached `evo_comp_book.pdf`, section 4.4.1, printed pp. 56-57 (PDF pp. 70-71),
describes real-valued Gaussian perturbation and uniform random-reset mutation.
These are the default operators. Section 4.4.2 describes self-adaptation, but the
old adaptive search also changes other EA components, so it is not in this study.

**Fitness qualification:** the assignment's default is final planar Euclidean
distance and it permits a clearly justified alternative. This project retains
its existing final 3D core-to-target-surface distance plus 100-point fall/off-terrain
penalty (1,000,000 for numerical failure). Report and justify this terrain-aware
choice explicitly. A resting core can remain above the surface target, so zero
fitness is not implied to be attainable. Success uses 0.1 m plus physical contact.

**Stopping qualification:** the brief recommends running to a plateau. The study
disables per-run patience to ensure equal fixed budgets across all conditions.
Use a pilot to choose a common sufficient budget; inspect the convergence curves.
If conditions are still improving, plan a larger common budget before the final
study. Do not claim convergence or a local optimum solely from a short plateau.
Resume preserves the original budget; a different budget requires a fresh study.

## Saved evidence

- `study.json`: frozen plan, source/model hashes, dependency versions, progress.
- `runs.csv`: one row per completed condition/seed, fitness, 3D distance,
  success/failure indicators, evaluation count and saved brain location.
- `summary.json`, `REPORT.md`: mean/sample SD and paired differences to random search.
- `convergence.png` / `.svg`: mean best fitness +/- sample SD across seeds,
  generation axis and matching evaluation-count axis.
- `diversity.png` / `.svg`: genotype spread and number of distinct surviving brains.
- `final_results.png` / `.svg`: individual paired seed results, mean/SD, target
  success and failure fractions. Grey lines join the same seed across conditions.
- `videos/`: MP4 and initial-frame PNG for **each run's best controller**, including
  the baseline. Numerically invalid controllers are explicitly marked as skipped;
  fallen but numerically valid controllers can still be shown.
- `runs/.../attempt_.../run/`: SQLite archive, JSON checkpoint with RNG state,
  best brain, history, trajectory and per-run diagnostic plots.
- Copies of both Python sources and the exact binary model for reproducibility.

Graphs use only seeds completed for **every** condition and baseline. Partial
studies identify missing repeats; they never silently pad stopped curves or rank
conditions with different budgets/seeds. SD is between-run spread, not a confidence
interval. No significance or global-optimum claims are made. One-seed smoke plots
have no SD band. Interpret these results only for this body, terrain and budget.

Resume checks source/model hashes and NumPy/MuJoCo versions. Changed code requires
a fresh study or use of the saved sources. Only worker count, session time cap and
video deferral may change on resume. Failed videos can be retried by resuming a
completed study; search results are not discarded.

## Quick verification

```powershell
.\.venv\Scripts\python.exe assignments/assignment_2/A2_template_2026.py --mutation-study --smoke --population 4 --generations 2 --duration 0.2 --seeds 42 43 --sigmas 0.05 0.15 --operators gaussian uniform_step uniform_reset --workers 2
.\.venv\Scripts\python.exe -m unittest discover -s assignments/assignment_2 -p 'test*_A2_2026.py' -q
```

Smoke outputs are explicitly labelled as pipeline checks, not research evidence.
For a single run instead of the study, use `--mutation-operator` with
`--search-mode standard`. Single-run video export also defaults on; `--view`
opens an optional interactive replay and `--no-video` disables export.

## Verified on 2 October 2026

All 48 regression tests passed; the six mutation-study tests were rerun after
final condition-name validation. A two-seed, six-condition smoke study completed
144 search evaluations, generated all three comparison figures (PNG/SVG), and
saved 12 MP4s with readable final frames. Exact initial chromosomes/fitness were
matched across conditions. Split/resumed runs matched uninterrupted populations
and RNG states for all three operators and random search. Resuming the completed
study generated no additional search attempts.

See `mutation_studies/smoke_final_20261002/REPORT.md` for that smoke check. The full
25-run research study has not been launched and there is no mutation winner yet.
