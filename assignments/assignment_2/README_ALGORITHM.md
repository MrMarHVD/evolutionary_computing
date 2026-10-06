# Assignment 2: how our robot-brain algorithm works

The current mutation-only experiment is documented in
[README_mutation_study.md](README_mutation_study.md). It fixes all non-mutation
settings, uses five paired seeds and random search, and saves comparison graphs
and videos. The guide below describes the general trainer, including legacy
search modes that are excluded from the controlled study. Videos now default
on and the interactive viewer requires `--view`.

## Short summary

- **What evolves:** 360 weights of a neural network controlling the fixed John
  Set snake. The body stays fixed; movement is learned through evolution.
- **Task:** approach the centre of OlympicArena's last platform, approximately
  `(6.07, 0, 0.505)` m, within 15 simulated seconds on reproducible terrain.
- **Fitness:** minimise final 3D core-to-target distance, with a 100-point
  fall/off-terrain penalty. Success additionally requires physical terrain
  contact and a final 3D gap of at most 0.10 m.
- **Evolution:** tournament selection, optional uniform crossover, Gaussian
  mutation and elitist survival. Defaults are 32 parents plus 32 children,
  with up to 150 offspring generations and plateau stopping.
- **Evidence:** a replayable brain and passing implementation checks establish
  correct execution. They do not establish good navigation. Final comparisons
  need at least five independent seeds, matched evaluation budgets and a baseline.
- **Review fixes, 2 October 2026:** numerical failures now produce usable failed-run
  records and count in failure statistics. A separate comparison plot/CSV shows
  cross-seed mean and sample SD. Existing fitness and training defaults are retained.


This guide explains the implementation in
[A2_template_2026.py](A2_template_2026.py). It describes what the program actually
does, why the main choices matter, and how to run and interpret an experiment.

We are using **neuroevolution**: an evolutionary algorithm searches for the
weights of a neural network that controls a simulated snake. Each candidate
brain gets a fresh attempt at the same task. Brains that finish closer to the
target are more likely to contribute to the next generation.

The robot's body stays fixed. What evolves is the collection of numbers inside
its controller.

## Contents

1. [The task and what can change](#1-the-task-and-what-can-change)
2. [The two loops in the program](#2-the-two-loops-in-the-program)
3. [What the brain observes](#3-what-the-brain-observes)
4. [How observations become movement](#4-how-observations-become-movement)
5. [How a brain is stored and initialised](#5-how-a-brain-is-stored-and-initialised)
6. [One generation of evolution](#6-one-generation-of-evolution)
7. [How a trial is simulated and scored](#7-how-a-trial-is-simulated-and-scored)
8. [The four search modes](#8-the-four-search-modes)
9. [Stopping rules and evaluation budgets](#9-stopping-rules-and-evaluation-budgets)
10. [Workers, seeds and reproducibility](#10-workers-seeds-and-reproducibility)
11. [Saved files, replay and continuation](#11-saved-files-replay-and-continuation)
12. [Commands to run the program](#12-commands-to-run-the-program)
13. [Reading the output](#13-reading-the-output)
14. [Designing a fair experiment](#14-designing-a-fair-experiment)
15. [Where to find each part in the code](#15-where-to-find-each-part-in-the-code)
16. [What the results do and do not establish](#16-what-the-results-do-and-do-not-establish)

## 1. The task and what can change

The task is to move the robot's core close to a fixed target within a fixed
amount of simulated time.

| Part | Current setup |
|---|---|
| Robot | Prebuilt `john_set.snake`, with eight actuated hinges |
| World | OlympicArena, regenerated with terrain seed 2026 |
| Starting XY | `(0, 0)` m |
| Starting orientation | 180 degrees of yaw; forward points toward world +X |
| Target XYZ | Centre of the top of `finish_end`, approximately `(6.07, 0, 0.505)` m |
| Trial duration | 15 simulated seconds |
| Physics and control | 0.002 s per step, or 500 Hz |
| Fitness | Final 3D core-to-target distance plus a fall/off-terrain penalty |
| Success | Final 3D gap at most 0.10 m, no fall, above terrain and in physical contact |
| Evolved values | 360 neural-network weights, including biases |

The nominal spawn is `[0, 0, 0.1]`. Collision correction adjusts its height;
the compiled model contains the actual starting pose. The target is derived
from the final platform's top surface when the world is built. Set
`TARGET_ON_FINAL_PLATFORM = False` to use a manual target for a different experiment.

The red pole and yellow ball mark the target. They cannot collide with the robot.
The brain receives target coordinates directly, without a camera or image recognition.

Body, terrain, target, architecture and objective stay fixed within a comparison.
Changing mutation changes the search; changing these task settings changes the
problem. The assignment's default metric is planar distance, but it permits a
different clearly justified metric. Our 3D and fall-aware metric distinguishes
the raised finish platform from falling underneath it. Describe that choice in
the report; historical planar results cannot be compared directly with it.

The assignment-local `_SeededOlympicArena` reproduces ARIEL's heightmap formula
with an explicit Perlin-noise seed. Independent search seeds therefore share
the same terrain. Replay uses each archive's exact saved model. The evolutionary
search is implemented through `ariel.ec`; it uses no black-box optimiser or CPG.

## 2. The two loops in the program

There are two different processes, running at different scales.

**The control loop** runs inside one simulation. A fixed brain repeatedly reads
the robot state and produces hinge commands. Its weights do not change during
that trial.

**The evolutionary loop** runs around many simulations. It compares brains,
creates new weight vectors, evaluates them, and chooses which brains survive.

```mermaid
flowchart TD
    A[Create an initial population of random brains] --> B[Simulate and score every initial brain]
    B --> C[Save generation 0]
    C --> D[Select parents from the surviving population]
    D --> E[Create children using crossover and mutation]
    E --> F[Evaluate each child in a fresh simulation]
    F --> G[Choose survivors from parents and children]
    G --> H[Save scores, best brain and checkpoint]
    H --> I{Stopping condition reached?}
    I -- No --> D
    I -- Yes --> J[Reload and verify the saved winner]
    J --> K[Summarise results and optionally replay]
```

Within each simulation, the smaller loop is:

```text
Observe the state -> calculate hinge commands -> advance physics -> repeat
```

A brain can react to changing observations without learning during the trial.
Learning here means changing which weight vectors exist in the population
between trials. We do not use backpropagation, gradient descent or a neural
network training dataset.

Useful terms:

| Term | Meaning in this project |
|---|---|
| Brain/controller | The neural network that produces hinge commands |
| Gene | One adjustable network weight or bias weight |
| Genotype/chromosome | The flat list of all 360 genes |
| Individual | A chromosome together with its score and stored information |
| Population | The group of individuals currently competing and reproducing |
| Evaluation | One complete attempt by one candidate brain |
| Generation | One round of creating, evaluating and selecting a batch of children |
| Fitness | The score used to rank brains; smaller is better here |

## 3. What the brain observes

`_controller_inputs()` builds a vector of 36 numbers at each control update.
One of these is a constant bias input, so there are 35 changing observation or
task/time features plus that constant.

| Input group | Count | What it tells the controller | Scaling in the code |
|---|---:|---|---|
| Core orientation | 4 | How the core is rotated, represented by a quaternion | Included in `qpos[3:]`, clipped to `[-pi, pi]` and divided by `pi` |
| Hinge angles | 8 | How the eight joints are currently bent | Same clipping/division as orientation |
| Velocities | 14 | Linear/angular motion of the floating base and motion of the eight hinges | `tanh(qvel / 5)` |
| Target displacement | 2 | The X/Y differences between the target and the current core position | `tanh((target_xy - core_xy) / 2)` |
| Clock features | 4 | Sine and cosine signals at 0.75 Hz and 1.5 Hz | Already bounded between -1 and 1 |
| Elapsed time | 1 | What fraction of the trial has passed | `clip(time / duration, 0, 1)` |
| Heading error | 2 | Whether the robot faces toward the target, and which way it needs to turn | Sine and cosine of the signed planar angle |
| Bias | 1 | A constant input that allows learned offsets | Always 1 |
| **Total** | **36** | | |

The quaternion is four numbers representing a rotation; it is not four angles.
Its components already lie between -1 and 1 for a unit quaternion. The current
code still divides them by `pi` together with the hinge angles. This is the
implemented scaling, not a claim that it is the only or best normalisation.

The exact order is:

```text
orientation (4), hinge angles (8), velocities (14), target XY gap (2),
sin(0.75 Hz), sin(1.5 Hz), cos(0.75 Hz), cos(1.5 Hz),
elapsed fraction, heading sine, heading cosine, constant 1
```

Keeping inputs bounded helps prevent a feature with a large numerical range
from dominating the network. It does not guarantee good control.

### Why include the target and heading?

The target displacement gives the controller information about where it should
go. It is expressed in world X/Y coordinates. The heading pair gives a more
direct signal about the turn needed to face the target.

For this snake, forward is core-local **-X**, away from the tail. When the target
is straight ahead, the heading features are approximately `(0, 1)`. Sine/cosine
avoid a sudden numerical jump between angles near -180 and +180 degrees. If
the planar direction is undefined, such as exactly at the target, the code
returns a finite neutral pair `(0, 1)`.

### Why include clocks and elapsed time?

Locomotion needs coordinated changes over time. The sine/cosine inputs give the
network repeating signals it can use to generate rhythmic commands. Evolution
still has to discover which combinations move this body effectively.

Elapsed time provides different information: it distinguishes the beginning
from the end of the trial, even when the repeating clocks are at the same phase.

These features do not prescribe which joint must bend in which direction. The
controller remains a neural network with evolved weights, rather than a CPG
controller or a manually specified snake gait.

## 4. How observations become movement

The network has one hidden layer of eight units and eight outputs, one for each
hinge. A hidden unit forms a weighted combination of the inputs and passes it
through `tanh`, a smooth function bounded between -1 and 1.

The forward pass is:

```text
inputs  = the 36-element observation vector
hidden  = tanh(inputs @ W1)
outputs = tanh(append(hidden, 1) @ W2)
actions = outputs * (pi / 2)
```

`@` means matrix multiplication: each unit adds up the incoming values multiplied
by their weights. The extra `1` before the output layer provides output biases.
The input vector's own constant `1` provides hidden-layer biases.

The weight matrices have these sizes:

| Matrix | Purpose | Shape | Number of weights |
|---|---|---|---:|
| `W1` | Observations and input bias to hidden units | `36 x 8` | 288 |
| `W2` | Hidden units and output bias to hinges | `9 x 8` | 72 |
| **Total** | | | **360** |

Biases allow a unit to produce a nonzero response even when its other inputs
are zero. They are already included in these matrices; there are no additional
bias arrays outside the 360-gene chromosome.

Each output is scaled to between `-pi/2` and `+pi/2` radians, or -90 to +90 degrees.
For example, a network output of `0.5` becomes a command of +45 degrees.

The program writes these as **direct desired-angle commands**:

```python
data.ctrl[:] = actions
```

The simulated actuators and physics determine how the joints actually move
toward those angles. Setting a desired angle does not instantly teleport a
joint into that position. We do not accumulate small output increments onto
the previous command.

The network has no recurrent hidden state. Its responses depend on the current
inputs, including velocities and time; the robot and physics still carry state
from one timestep to the next.

## 5. How a brain is stored and initialised

Evolution works with a flat vector of 360 real numbers. `_decode()` reshapes
the first 288 into `W1` and the remaining 72 into `W2`.

```text
360-gene chromosome -> W1: 36 x 8, W2: 9 x 8 -> working controller
```

On a fresh run, every gene is drawn independently from a normal distribution
with mean 0 and standard deviation 0.5. Values are then clipped to `[-5, 5]`.
The default initial population contains 32 such brains.

The weight bound of 5 is different from the hinge-angle limit. Weights influence
the computation inside the network; the final `tanh` and angle scaling bound
the commands sent to the robot.

All initial brains are evaluated before any parent selection happens. Their
results are recorded as **generation 0**. They already consume an evaluation
budget: with population 32, generation 0 costs 32 evaluations.

`_decode()` rejects chromosomes with the wrong length or non-finite values,
such as `NaN`, instead of silently reshaping an incompatible brain.

## 6. One generation of evolution

The standard algorithm is a real-valued evolutionary algorithm with tournament
parent selection, uniform crossover, Gaussian mutation and elitist survival.
ARIEL supplies the population/individual objects, operation pipeline and
database storage. Our assignment code implements the search operators.

### Step 1: select parents using tournaments

For each child, the algorithm chooses two parents. To choose one parent, it
randomly draws three indices from the current surviving population and takes
the individual with the **smallest distance** among those draws.

The draws are with replacement, so an individual can appear more than once.
The two selected parents can also be the same individual. Tournament size is
controlled by `--tournament`, with a default of 3.

This gives better brains more opportunities to reproduce while allowing other
brains to contribute. Larger tournaments generally concentrate selection more
strongly on the best individuals.

All children in a generation use the population that existed before breeding
started. A newly created child does not become a parent within that same round.

### Step 2: optionally combine the parents

The child begins as a copy of the first parent's chromosome. In standard mode,
uniform crossover happens with probability 0.2 per child.

When crossover happens, each gene independently has a 50% chance of being
replaced with the corresponding gene from the second parent. This copies
weights; it does not average them or merge whole hidden neurons.

For a miniature four-gene example:

```text
Parent A:                [ 0.2, -0.4,  0.6,  0.1 ]
Parent B:                [ 0.8,  0.3, -0.2, -0.5 ]
Take genes from B here:  [  no,  yes,    no,  yes ]
Child before mutation:   [ 0.2,  0.3,  0.6, -0.5 ]
```

Combining weights can produce useful combinations, but it can also break
coordinated behaviour. It is an operator to investigate, not an automatic gain.

### Step 3: mutate the child

Every child then goes through mutation. In standard mode, each gene independently
has probability 0.1 of receiving additive Gaussian noise:

```text
if this gene is selected:
    new_weight = old_weight + noise

noise has mean 0 and standard deviation mutation_sigma
```

The default `mutation_sigma` is 0.15. This is the scale of the random changes,
not a maximum change size.

With 360 genes and mutation rate 0.1, the expected number selected per child is
`360 * 0.1 = 36`. The actual number varies. Mutation rate 0.1 does **not** mean
that only 10% of children are mutated, nor that exactly 36 genes always change.

All child weights are clipped back to `[-5, 5]` after mutation. The child becomes
a new `Individual` without a fitness value; it must earn its own score. It does
not inherit its parent's fitness.

### Step 4: evaluate the new children

The algorithm creates and evaluates as many children as the population size:
32 by default. Existing parents retain their scores because every evaluation
uses the same deterministic task and initial state.

There is no general cache that skips identical newly created chromosomes.
Even a newly created copy is evaluated and counted. Focused mode reduces one
source of unchanged copies, as explained below.

### Step 5: select survivors from parents and children

Standard and mixed modes combine the 32 parents and 32 children, sort all 64
by ascending fitness, and keep the best 32.

This is called **(mu + lambda) survival**: `mu` is the number of parents and
`lambda` is the number of children. Here both equal the population size.
Parents compete directly with their children. A parent can survive for many
generations if it remains competitive.

For example, if a good parent scores 0.40 m and its child scores 0.65 m, mutation
has made that child worse. The parent is still eligible to survive. If another
child scores 0.30 m, it can improve the population's best result.

Keeping the best available individual is **elitism**. It means the best saved
fitness cannot get worse from one completed generation to the next in this
deterministic search. It does not mean that every child improves.

Rejected individuals are marked dead, then stored with the rest of the
generation in SQLite. They remain available for analysis even though they no
longer reproduce.

### Step 6: record and repeat

After survivor selection, the program saves population statistics, the best
brain, and a checkpoint containing the surviving population and random-number
generator state. It then checks whether to stop or create the next generation.

## 7. How a trial is simulated and scored

`_simulate()` applies the same evaluation to every candidate:

1. Clear the previous controller, reset all simulation data, and initialise state.
2. Attach the candidate's neural network as the physics-step controller.
3. Advance the requested number of physics steps, checking numerical validity
   and falls every 50 steps.
4. Read the final core position, inspect terrain support, and calculate fitness.
5. Clear the callback in `finally`, including after a failed trial.

A default trial contains `15 / 0.002 = 7,500` physics steps. The callback runs
every physics step. Chunks of 50 only batch monitoring and trajectory samples;
they do not lower the control rate to 10 Hz.

The current objective is fitness version 2:

```text
distance_3d = sqrt((final_x - target_x)^2
                 + (final_y - target_y)^2
                 + (final_z - target_z)^2)

fitness = distance_3d + (100 if fallen or not on_terrain else 0)
```

`fitness_function()` calculates the distance; `_score_state()` adds the penalty.
The target is on the platform surface. A supported core exactly above its
centre at Z = 0.580 m has a 0.075 m gap to the target at Z = 0.505 m. Zero is the
mathematical distance minimum; this does not imply that zero is physically
attainable by the robot's core.

Terrain support has an explicit meaning in the code:

- A downward ray under the core finds static collidable terrain. Robot geometry
  and the decorative markers are excluded.
- Core clearance must be between -0.02 m and +0.25 m relative to that surface.
- `fallen` means core Z has dropped more than 0.25 m below its actual spawn Z.
  Such trials stop at the next monitoring boundary, instead of continuing to
  drift during free fall. Viewer and video use the same fall-check boundaries.
- `terrain_contact` records an actual robot/terrain contact within 0.001 m.

`on_terrain` is a clearance check; it does not itself guarantee physical contact.
An airborne core close to the terrain can pass it. The 100-point penalty follows
the formula above, while **success additionally requires `terrain_contact`**.
The success test also requires numerical validity, no fall, and fitness at most
`--target-radius`, default 0.10 m. This checks final support, rather than a
sustained hold or every body segment resting on the finish platform.

Only the final position is rewarded. Visiting the target and then moving away
can score poorly. There are no explicit energy, smoothness, path-length or
arrival-speed terms. The core is the reference point, and initial position is
unused by the distance function because the experiment fixes the start.
Success is a reported label; it does not stop evolution or end a valid trial early.

Numerical failures receive `BAD_FITNESS = 1,000,000`, separately from physical
task failures. Checks cover finite actions/state, MuJoCo warnings and the expected
simulated clock. Missing or non-finite measurements are stored as JSON `null`,
never NaN/Infinity. A fully invalid population still saves history, checkpoints
and a failed result. Verification must reproduce both score and validity:
`replay_verified: true` can therefore describe a reproducible failure, and must
be read together with `valid` and `reached_target`. Invalid brains skip animation.

## 8. The four search modes

`--search-mode` chooses a variation strategy. It does not change the robot,
network, simulation duration or fitness. The default remains `standard`.

### Standard

Each child uses the configured mutation rate and sigma. Crossover is allowed
for every child, with the configured probability.

With the defaults, this means 0.1 mutation probability per gene, sigma 0.15,
and a 0.2 probability of attempting crossover per child.

### Mixed

Each child independently draws one of three mutation scales:

| Chance of drawing this category | Sigma multiplier | Sigma when base sigma is 0.15 | Crossover |
|---:|---:|---:|---|
| 60% | 0.2 | 0.03 | Disabled |
| 30% | 1.0 | 0.15 | Allowed with the configured probability |
| 10% | 3.0 | 0.45 | Allowed with the configured probability |

All three categories still use the configured per-gene mutation probability.
Fine mutations therefore make smaller changes to about 36 genes on average
at the defaults; they do not necessarily change fewer genes.

The intended tradeoff is to spend many attempts refining a brain while keeping
some larger changes that could discover a different movement pattern. The
60/30/10 split is probabilistic, not a guaranteed allocation in every generation.

With crossover rate 0.2, the overall probability of attempting crossover in
mixed mode is `0.4 * 0.2 = 0.08`, because only the normal/large categories allow
it. This matters when interpreting comparisons with standard mode.

### Focused

Focused mode uses the same 60/30/10 category probabilities, with three changes:

1. The fine category selects **one random gene** to mutate at 0.2 times the
   base sigma. It skips crossover. The per-gene mutation-rate setting governs
   the normal/large categories, not this single-gene fine category.
2. If the completed child is exactly identical to its first parent, the code
   forces a small change to one gene. This handles an empty mutation mask or
   changes erased by clipping. It does not guarantee that the child differs
   from every other individual ever evaluated.
3. During survival, the best copy of each distinct chromosome is considered
   before duplicate copies. If there are too few distinct chromosomes to fill
   the population, duplicates fill the remaining slots.

Normal and large mutations use the same rules as mixed mode. Focused mode
requires positive mutation rate and sigma.

Duplicate handling uses exact chromosome equality, not similarity of gaits.
Two different chromosomes can still produce almost the same movement. The
best brain remains protected, although a worse distinct brain can take a slot
that would otherwise go to another copy of a better one.

Focused mode is experimental. Its mechanisms are implemented and tested;
better performance on this task requires a comparison over independent seeds.
It provides an option for testing a search hypothesis.

### Adaptive

`--search-mode adaptive` adds **self-adaptive mutation strength**. Each brain
can carry a positive `mutation_sigma` tag alongside its 360 neural weights.
The tag controls breeding only: it is not a neural input, does not alter the
network size, and does not change during a simulation.

For each child:

1. Select one parent using the same fitness tournament as standard search.
2. Inherit its mutation sigma; an initial parent without this tag uses the
   configured `--mutation-sigma` (0.15 by default).
3. Draw one standard normal random value `z` and calculate:

   ```text
   child_sigma = clip(parent_sigma * exp(0.7 * z), 0.002, 0.5)
   ```

4. Copy the parent's weights. Select each weight with `--mutation-rate`
   probability and add independent mean-zero Gaussian noise with standard
   deviation `child_sigma`. If the mask is empty, select one random weight.
5. Clip weights to the usual `[-5, 5]` range. If clipping leaves the entire
   chromosome unchanged, move one weight inward by `child_sigma`.
6. Evaluate the child and store its sigma with its fitness and final position.
   Keep the best distinct chromosomes first, filling spare slots with duplicates
   only when needed, as in focused mode.

For example, a parent sigma of 0.15 and `z = -1` give a child sigma of about
0.0745. A child that survives with this smaller step can pass it to its own
children. Positive `z` values also allow larger steps. Thus selection indirectly
favours mutation strengths that accompany successful controllers; there is no
rule that sigma must shrink at every generation.

Adaptive mode skips crossover, regardless of `--crossover-rate`, to preserve
the coordinated weight combinations in each parent. `ADAPTIVE_SIGMA_MIN`,
`ADAPTIVE_SIGMA_MAX` and `ADAPTIVE_SIGMA_TAU` control the bounds and log-normal
change strength in the settings section. The per-gene mutation probability
still applies; this is not the focused mode's single-weight mutation strategy.

The motivation is that a fixed noise scale may be too disruptive when refining
an existing gait. This remains a search heuristic, not a guarantee of progress.
Comparing it with standard search tests a package of changes: inherited sigma,
no crossover, guaranteed mutation and duplicate-aware survival.
Adaptive mode remains experimental. Its mechanisms are implemented and tested;
this does not establish that it beats standard search on the current task.

## 9. Stopping rules and evaluation budgets

Training stops at the first applicable condition:

- The maximum number of additional offspring generations has been completed.
- The configured number of generations has passed without a sufficiently
  large improvement in the best score.
- An optional `--max-minutes` training budget has expired. The current generation
  finishes before checkpointing; final plots and winner verification add some time.

For a fresh run, the built-in defaults are `--generations 150`, `--patience 30`
and `--min-improvement 0.0001`.

These defaults are a starting configuration, not a measured optimum. A larger
budget allows more search; it does not improve the scoring of each candidate.
The tuner can also impose a wall-clock deadline and pause at a completed
generation boundary, with population and RNG state already checkpointed.

The improvement check is exactly:

```text
new_best < previous_significant_best - min_improvement
```

When that condition holds, the program records a new reference score and
resets the patience counter. Improvements must be **greater than** the threshold.
The reference is the last significant improvement, so small improvements can
accumulate before crossing it.

For example, starting from a reference of 1.00000 m with threshold 0.0001 m:
a new best of 0.99995 m does not reset patience; a later best of 0.99989 m does.
Every best result is still saved, even if its improvement is too small to
reset patience.

`--patience 0` disables plateau stopping. The generation limit still applies.
A plateau describes this search's recent progress; it is not proof that no
better brain exists.

### How many simulations does a run use?

For a fresh run with population size `P` and `G` completed offspring generations:

```text
training evaluations = P + G * P = P * (G + 1)
```

The extra `1` is generation 0, the initial population.

| Example | Maximum training evaluations |
|---|---:|
| Previous default: 32 individuals, 60 offspring generations | `32 * 61 = 1,952` |
| Current default: 32 individuals, 150 offspring generations | `32 * 151 = 4,832` |
| Continue an existing population of 32 for 100 more generations | `32 * 100 = 3,200` new evaluations |

Early stopping can reduce these counts. Rechecking the saved winner, verifying
a resume/replay, and producing a video are extra simulations outside the
evolutionary search budget.

The 15-second trial duration is **simulated** time, not wall-clock time. The
amount of real time depends on the laptop, worker count and simulation cost.

## 10. Workers, seeds and reproducibility

### What a worker does

A worker evaluates candidate brains. Each process loads the archived physics
model once, reuses its model/data objects, and resets the data before every trial.
With ten workers, up to ten candidate simulations can be evaluated concurrently.

Evolution waits for the generation's evaluations to finish before selecting
survivors. More workers change the scheduling and speed; they do not increase
the population, generation count or evaluation budget.

The implementation uses separate **processes**, rather than threads, because
MuJoCo's control callback is global within a process. Each worker therefore has
its own callback and simulation state. Results are matched to candidates in
submission order, not in whichever order workers finish.

By default, the program uses the smaller of 4 and the detected CPU count.
The examples below select 4 workers; the appropriate count depends on available
CPU capacity and measured runtime. Workers beyond the available population have
no additional candidates to process in that generation.

Multiple seeds run sequentially within one invocation. The worker pool
parallelises candidates within each run; it does not start all seed runs at once.

### What a seed controls

Each fresh run creates a NumPy random-number generator from its seed. This
controls initial chromosomes, tournament draws, crossover masks, mutation
masks/noise, adaptive sigma draws, and the category draws in mixed/focused modes.
ARIEL's own random generator is also seeded.

`TERRAIN_SEED` independently controls the regenerated OlympicArena heightmap.
All search seeds and methods must use the same terrain seed for paired
comparisons. Keep `WORLD_KWARGS["load_precompiled"] = False` to generate that
seeded terrain; a precompiled world uses its cached geometry instead. Other
world classes may need their own seed handling.

Different seeds give independent search attempts, while the task stays the
same. Running seed 42 again with the same settings repeats an experiment;
resuming seed 42 continues it. Neither gives a new independent replicate.

The program records the model, settings, source checksum and software versions.
Exact continuation also restores the NumPy RNG state. Reproducibility should
be checked using the same compatible code, model and software environment;
the code does not promise identical physics across arbitrary version changes.

## 11. Saved files, replay and continuation

Output is stored beside the assignment unless `--output-dir` selects another
parent folder. Each invocation reserves a new numbered directory and preserves
earlier runs:

```text
assignments/assignment_2/
    latest_best.json
    snake_olympic_arena_9/          # illustrative next number
        model.mjb
        source_snapshot.py
        latest_best.json
        convergence.png           # each run's best fitness by generation
        comparison.png            # cross-seed mean with sample SD
        comparison.csv            # exact values plotted in comparison.png
        summary.json
        ea_seed_42/
            config.json
            evolution.sqlite
            history.csv
            best_brain.json
            checkpoint.json
            fitness.png
            diversity.png
            trajectory.png
            trajectory.csv
            result.json
        random_seed_42/            # when --baseline is requested
```

Additional seeds have their own subfolders. Video export adds `best_replay.mp4`
and a starting-frame PNG to the chosen winner's directory.

`evolution.sqlite` stores every evaluated chromosome, including rejected candidates.
`history.csv` describes survivors after selection. `config.json` records settings,
task/controller versions, source/model hashes and software versions.
`checkpoint.json` saves the whole surviving population, RNG state, evaluation
counter and patience state after each completed generation. `best_brain.json`
stores the selected chromosome, measurements and numerical validity.
`result.json` separates numerical validity, target success and replay verification.
`summary.json` reports per-method mean/sample SD, success rate, validity rate and
failure rate. Failure includes numerical invalidity, falling or failing the
terrain-clearance check; a valid non-reaching snake need not be a failure.

JSON snapshots are replaced atomically. Recovery starts from the last complete
checkpoint; unfinished evaluations may be repeated. `latest_best.json` selects
the winner of the latest completed invocation, restricted to the requested
algorithm, with ties resolved by the smaller seed. It is not an all-time leaderboard.

### Replay

Replay checks the archived model checksum, restores its target and duration,
and verifies the saved score and validity before opening the viewer. Keep the
entire batch folder with a brain. A moved archive can be selected by explicit
brain or batch-folder path; old absolute latest pointers may need an explicit path.

Schema-3 brains from older objectives may be replayed diagnostically: their
stored score is labelled obsolete and the current score is printed. The archive
is not rewritten. Schema-1/2 brains use incompatible network inputs and are rejected.

### Resume

Resume restores the **whole population**, not just the winner. Current
checkpoints restore the RNG and patience state. Keeping compatible settings
reproduces an uninterrupted run, including adaptive mutation-sigma tags.
Every continuation writes a new batch, preserving the source archive.

The fitness settings must match exactly: older planar objectives require fresh
training rather than reuse of incomparable scores. Body, network, population
size, duration and supported software versions must also match. The archived
physics and target define the continued task even if today's defaults changed.

Search settings are inherited unless explicitly overridden. Changing search mode,
mutation rate/sigma, crossover or tournament size resets patience. Switching into
adaptive mode or explicitly changing its sigma configuration resets strategy
tags while retaining the neural weights. Unchanged resumes retain patience;
use a larger patience or `--patience 0` if more search after a plateau is intended.

`--generations` is an additional budget on resume. `evaluations` is cumulative;
`new_evaluations` excludes inherited work. Continuation is not an independent
replicate. Its SQLite file contains inherited survivors plus new evaluations;
earlier rejected candidates remain in the source archive.

Legacy archives without saved RNG state can supply a seeded population warm
start when otherwise compatible, but cannot reproduce the original random
sequence. The program labels that distinction explicitly.

## 12. Commands to run the program

Run the following commands from the `evolutionary_computing` repository folder.
The existing project environment must be installed. If `.venv` needs restoring,
use `uv sync --cache-dir .uv-cache --locked` first.

`--cache-dir .uv-cache` puts uv's package cache in a local folder. It is separate
from the `.venv` Python environment and has no effect on fitness or evolution.
`--locked` requires the dependency lockfile to remain consistent with the project.
Neither uv option is a setting of the evolutionary algorithm.

### Start fresh training

```powershell
uv run --cache-dir .uv-cache --locked python assignments/assignment_2/A2_template_2026.py --workers 4
```

This uses the updated defaults of at most 150 generations and patience 30. It
trains one fresh seed-42 population using standard search in the fixed-seed
OlympicArena and opens the winning replay afterward. Add `--no-view` to skip
the viewer, or `--video` to save an MP4.

With the file settings `REPLAY = None` and `RESUME = None`, omitting both flags
starts fresh training; it does not automatically build on a saved brain.

### Replay the latest completed default-output run

```powershell
uv run --cache-dir .uv-cache --locked python assignments/assignment_2/A2_template_2026.py --replay
```

Add `--no-view` to verify the saved score without opening a window. Add `--video`
to export the replay. Use `--replay "PATH_TO_BEST_BRAIN_JSON"` to select a specific
brain, replacing the quoted placeholder with its actual file path.

### Continue the latest completed population

This requires matching current fitness settings. Older objectives need fresh training.

```powershell
uv run --cache-dir .uv-cache --locked python assignments/assignment_2/A2_template_2026.py --resume --workers 4 --generations 100 --patience 0
```

Here `100` means **100 additional generations**. Plateau stopping is explicitly
disabled in this example. To recover an interrupted run, use
`--resume "PATH_TO_SEED_DIRECTORY"` or its `checkpoint.json` path instead of
relying on the latest completed-batch pointer.

### Run five seeds and a matched random-search baseline

```powershell
uv run --cache-dir .uv-cache --locked python assignments/assignment_2/A2_template_2026.py --workers 4 --seeds 42 43 44 45 46 --generations 60 --patience 0 --baseline --no-view
```

This uses the same fixed maximum budget for each seed. It runs an EA and a
matched random baseline for each seed, so allow time for ten runs.

### Select another mode or separate output folder

Append `--search-mode mixed`, `--search-mode focused` or `--search-mode adaptive`
to a training command. Standard is the default. Append, for example:

```text
--output-dir assignments/assignment_2/__data__/my_experiment
```

Each custom output folder gets its own latest pointer. Bare `--replay` and
`--resume` still look in the default output folder; use an explicit brain/run
path to select results from a custom folder.

### Inspect settings or run the implementation checks

```powershell
uv run --cache-dir .uv-cache --locked python assignments/assignment_2/A2_template_2026.py --help
uv run --cache-dir .uv-cache --locked python -m unittest discover -s assignments/assignment_2 -p 'test*_A2_2026.py' -v
```

The tests cover controller dimensions/actions, heading, fitness, reset behaviour,
invalid simulations, variation, survival, archive counts, baseline pairing,
exact continuation for all four modes, reproducible terrain, inherited adaptive
sigmas, archived-world replay, failed-run handling and cross-seed statistics.

### Main built-in defaults

All editable settings are grouped in **EDITABLE SETTINGS**, immediately after
the imports in `A2_template_2026.py`. Each variable has a comment explaining its
purpose and units. Change these values before starting Python; for example,
`POPULATION_SIZE = 64`, `SEEDS = [42, 43, 44, 45, 46]`, or `NO_VIEW = True`.
The corresponding command-line options still work and take precedence over
these defaults. Resume still inherits saved experiment settings unless an
explicit command-line option overrides them.

`BODY_FACTORY` and `WORLD_FACTORY` select constructors without calling them
(for example, `snake` and `SimpleFlatWorld`); `WORLD_KWARGS` contains the chosen
world's constructor options. Import any additional body/world before selecting
it. Check `SPAWN_ROTATION` and `FORWARD_X_SIGN` when changing bodies: the default
heading feature assumes forward is core-local -X. `TERRAIN_SEED` now seeds
regenerated OlympicArena terrain; it is separate from the search seeds in `SEEDS`.

The same section also contains controller scaling/clocks, initial weights,
mixed/focused mutation settings, output paths, replay/video options, target
markers and camera/plot styling. `MODE` only controls direct calls to
`run_experiment()`; `REPLAY` and `RESUME` select the normal script workflow.
Both default to `None` for fresh training; either accepts `"latest"` or a saved
path. Keep controller settings compatible with saved brains: replay/resume
checks their values as well as network dimensions. The explanations and numeric
controller examples in this README still describe the 36-input, 360-weight
network. The world and training budget reflect the updated defaults.

These are defaults for a **fresh** run, before explicit options or inherited
resume settings are applied.

| Option | Default | Meaning |
|---|---:|---|
| `--population` | 32 | Survivors retained and children generated each generation |
| `--generations` | 150 | Maximum offspring generations; additional generations on resume |
| `--max-minutes` | None | Total training time budget across seeds; stops at a generation boundary |
| `--seeds` | 42 | Seed or list of independent fresh-run seeds |
| `--duration` | 15.0 | Simulated seconds per trial |
| `--workers` | `min(4, detected CPU count)` | Concurrent evaluation processes |
| `--tournament` | 3 | Number of draws per parent-selection tournament |
| `--crossover-rate` | 0.2 | Chance of crossover when the selected mode/category permits it |
| `--mutation-rate` | 0.1 | Per-gene mutation probability, except focused fine mutations |
| `--mutation-sigma` | 0.15 | Base standard deviation of mutation noise |
| `--search-mode` | `standard` | Standard, mixed, focused or adaptive search |
| `--patience` | 30 | Generations allowed since a significant improvement; 0 disables |
| `--min-improvement` | 0.0001 | Threshold in metres for resetting patience |
| `--target-radius` | 0.1 | Maximum final distance labelled successful |
| `--algorithm` | `ea` | Evolutionary algorithm; `random` selects standalone random search |
| `--parameter-file` | None | Load validated tuner settings for a fresh run; explicit flags override |

## 13. Reading the output

A log line describes a completed generation:

```text
ea seed=42 generation=  2 evaluations=   12 best_fitness=6.07970 mean_fitness=6.08037 best_3d=6.07970 m valid=100% fallen=0%
```

This particular line comes from the 2 October 2026 smoke run: population 4,
two offspring generations, and **0.1-second trials**. It verifies the pipeline,
not locomotion over the normal 15-second task.

`best`, `mean`, `worst`, `fitness_std` and status fractions in `history.csv`
describe the surviving population after selection. They do not describe all
attempted children. The SQLite archive contains rejected candidates too.

Additional measurements include:

- `valid_fraction`: fraction of numerically valid survivors. Valid does not mean successful.
- `fallen_fraction` and `on_terrain_fraction`: physical task diagnostics.
- `target_fraction`: fraction meeting the complete final success test.
- `best_distance_3d` and `best_core_height`: measured geometry, separate from penalties.
- `mean_gene_std`: mean per-weight SD across survivor chromosomes.
- `unique_genotypes`: exact distinct chromosome count.
- `mean_mutation_sigma`: inherited mutation strength, in adaptive EA runs.

Numerically different chromosomes need not produce different useful gaits.
These diversity statistics diagnose the population; they add no fitness bonus.

`convergence.png` and each seed's `fitness.png` retain individual best-fitness
curves. `comparison.png` is separate: it plots **the mean of each independent
run's best fitness**, with a band of plus/minus one **sample SD** (`ddof=1`).
This is neither the within-population mean nor a confidence interval.

The comparison uses only evaluation budgets observed in every included run.
If a seed stopped early, later generations are omitted from the aggregate;
they remain visible in individual curves. The cohort stays fixed, so the mean
does not improve merely because weaker runs drop out. With one seed, SD is
undefined and no band is shown. `comparison.csv` exposes generation, budget,
sample size, mean and sample SD for every plotted point. All methods in this
generation plot must have matching budget-to-generation relationships.

### A verified current-objective archive

On 2 October 2026, a headless replay of
`snake_olympic_arena_4/ea_seed_42/best_brain.json` reproduced:

| Value | Interpretation |
|---|---|
| Fitness / 3D gap `4.9735378425` | No failure penalty; approximately 4.97 m from the final-platform target |
| Generation `87` | 87 offspring generations after generation 0 |
| Evaluations `5632` | Population 64: `64 * (87 + 1)` |
| Duration `15.0` s | Full evaluation duration |
| `valid: true`, `fallen: false`, `on_terrain: true` | Numerically valid, terrain-supported final pose |
| `reached_target: false` | Outside the 0.10 m success radius |

This is a dated, verified archive example. It is not a claim that it is the
best controller in every folder, nor evidence that the current task is solved.
Historical planar-distance pilots are documented separately in
`REFINEMENT_RESULTS.md`; do not mix their scores with this objective.

### Implementation review on 2 October 2026

The review checked the visible assignment requirements, selection direction,
offspring freshness, fixed terrain, neural dimensions/actions, reset behaviour,
scoring, baseline budgets and saved-state recovery. The following gaps were fixed:

- A fully invalid population previously raised a replay-mismatch exception even
  when its saved failure score reproduced exactly. Verification now compares
  score **and validity**, and produces a useful failed-run result.
- Non-finite status values could break strict JSON saving. Unavailable values
  are represented as `null` and cannot count as target success.
- Tuner failure rates now include numerical failure, even if the last recorded
  pose still passes the terrain-clearance check. Distance summaries use valid
  trials only; penalised fitness averages keep every trial.
- The required mean/spread across independent runs is now available in the
  separate `comparison.png` and `comparison.csv` outputs.

The fitness version, controller schema and default training parameters are unchanged.

Validation passed all 42 controller/tuner regression tests. A five-seed,
two-worker smoke run completed 120 search evaluations (12 per seed per method),
with matching EA/random initial populations and budgets. Every saved winner was
replayed, and the comparison CSV was checked against a fresh calculation from
the individual histories. The short 0.1-second trials establish execution and
accounting, rather than target-reaching performance.

## 14. Designing a fair experiment

The assignment calls for investigating an aspect of the EA, with at least five
independent repeats, a baseline and statistics. Code that produces movement
is the starting point; the experiment needs a clear question.

For example: **How does Gaussian mutation strength affect final distance at an
equal number of evaluations?** Compare sigma values while holding population,
world, body, inputs, duration, selection and other operators fixed.

### What the random-search baseline does

Random search starts from the same seeded initial population as the matching EA.
After that, each new candidate is an independent draw from the original
mean-0, standard-deviation-0.5 weight distribution, clipped to `[-5, 5]`.
Previously good candidates do not influence where its new samples are drawn.

The program retains good random samples to report best-so-far results, but that
does not turn its sampling into evolution. It uses neither tournament-selected
parent genes nor crossover/mutation to produce the next random candidates.

With `--baseline`, the baseline gets exactly the EA's actual training evaluation
count for that seed, even if the EA stopped early. Baseline patience is disabled
so it uses the full matched count. A resumed population cannot be combined with
this fresh-run baseline option, because it already contains inherited search work.

### Controls that matter

- Keep the task, simulation duration, controller inputs/architecture and initial
  distribution identical across the methods being compared.
- Compare equal candidate-evaluation budgets. Equal generation counts are not
  equal budgets if population sizes differ.
- Use the same seed list for paired comparisons and at least five independent
  seeds for the final configurations. A continuation is not another replicate.
- Report mean and spread across runs, plus success rate if useful. Choosing
  only the luckiest run can give a misleading picture of a method.
- For a mutation-only comparison of standard and mixed modes, set
  `--crossover-rate 0` in both. Otherwise their effective crossover frequencies
  differ as well as their mutation scales.
- Focused mode also changes duplicate handling during survival. Comparing the
  whole focused package to standard search does not isolate one mutation change.
- State whether a run stopped on a plateau or exhausted its generation cap.
  A short pilot need not be a converged result.

The historical [refinement pilot](REFINEMENT_RESULTS.md) documents small comparisons on older
versions of the task. Those results
are evidence for those particular budgets, seeds and versions, not a guarantee
that one mode always wins. New experiments should keep their own configs and
results rather than treating the README as a live results database.

## 15. Where to find each part in the code

All functions below are in [A2_template_2026.py](A2_template_2026.py).

| Function | Responsibility |
|---|---|
| `main()` | Build the current model, display network dimensions and dispatch the command-line workflow |
| `build_world()` / `build_robot()` | Define the fixed task environment and snake body |
| `_controller_inputs()` / `_heading_features()` | Build the state, target and timing features |
| `nn_controller()` | Perform the network forward pass and produce desired hinge angles |
| `_decode()` | Turn one flat chromosome into the two weight matrices |
| `fitness_function()` / `_score_state()` | Calculate 3D distance and add physical-failure penalties |
| `_task_status()` / `_reached_target()` | Measure terrain support and evaluate final success |
| `_simulate()` | Reset and evaluate one controller consistently |
| `_worker_init()` / `_worker_evaluate()` | Reuse a model in each evaluation process and score candidates |
| `_evaluate_population()` | Evaluate individuals whose fitness has not yet been assigned |
| `_breed()` | Create the next batch of evolutionary or random-search candidates |
| `_survive()` | Select the next surviving population and mark rejected candidates |
| `_train()` | Manage one run's generations, stopping, checkpoints and winner verification |
| `_record_generation()` / `_save_checkpoint()` | Save statistics, the best brain and continuation state |
| `_load_brain()` / `_archived_model()` | Validate a saved brain and load its original physics |
| `_continuation_state()` | Read a current checkpoint or recover a legacy population |
| `_plot_histories()` | Plot individual best-fitness curves |
| `_aggregate_histories()` / `_plot_comparison()` | Calculate and plot mean/sample SD at common observed budgets |
| `_interactive_replay()` / `_export_video()` | Show or save the winning behaviour |
| `_cli()` | Parse options and orchestrate fresh runs, baselines, continuation and replay |

The evolutionary operations supplied to ARIEL are, in order:

```text
_breed -> _evaluate_population -> _survive
```

`_train()` calls `ea.step()` to execute them and persist the generation. The
training path uses `_worker_evaluate()` and `_simulate()` directly; it does not
rebuild a world through `run_experiment()` for each individual.

Some original template comments still describe a random-only demonstration or
say "YOUR JOB". They are retained as historical template guidance. The actual
`main() -> _cli() -> _train()` path already implements evolution.

## 16. What the results do and do not establish

A lower fitness is a better outcome under this fixed scoring rule. When both
trials avoid physical-failure penalties, it also means a smaller final 3D gap. Reproducing a saved score confirms that the archived
controller can be evaluated consistently in the supported setup.

It does not by itself establish that the robot:

- Reaches arbitrary targets or recovers from unexpected pushes.
- Works from new starting orientations, on other terrain, or for another duration.
- Moves smoothly, efficiently or safely; these are not explicit fitness terms.
- Stays at the target after the trial ends.
- Has found the best possible brain, even if progress has plateaued.

Our current work is to evolve and compare controllers for a clearly defined
fixed task. Successful target-reaching, reliable performance across seeds, and
generalisation to new conditions are separate outcomes that need their own
evidence.
