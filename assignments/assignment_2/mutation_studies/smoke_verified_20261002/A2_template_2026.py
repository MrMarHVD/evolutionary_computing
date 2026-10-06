"""Assignment 2: evolve a neural controller for the fixed John Set snake.

Use --mutation-study for the controlled five-seed mutation comparison, baseline,
plots and videos (see README_mutation_study.md). Without it, run one configurable
training experiment, or use --resume / --replay for an archived experiment.

The EA uses ariel.ec for population storage and operations. Fitness is final
3D target distance with terrain/fall penalties. The body and arena stay fixed.
Videos are saved by default; --view explicitly opens the interactive viewer.
"""

# Standard library
import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Literal

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import snake
from ariel.ec import set_seed
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.simulation.environments import BaseWorld
from ariel.simulation.environments import OlympicArena
from ariel.utils.noise_gen import PerlinNoise
from ariel.utils.renderers import single_frame_renderer

# Type aliases
type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]

# ============================================================================ #
#  EDITABLE SETTINGS
# ============================================================================ #
# Edit these before launching Python. Command-line options override their defaults.
# On --resume, saved run settings are inherited unless explicitly overridden by a
# command-line option. Keep the task/controller settings fixed when replaying a brain.
# Numbers that define the algorithm's structure (e.g. two parents, XY coordinates,
# bias inputs and file schema versions) stay with the code that uses them.

# --- Task: randomness, body, world and starting pose --- #
SEED = 42  # Seed for a single run and the direct random-controller demo.
SEEDS = [SEED]  # Independent training seeds; e.g. [42, 43, 44, 45, 46] (--seeds).
BODY_FACTORY = snake  # Body constructor, without (); e.g. snake or gecko.
WORLD_FACTORY = OlympicArena  # World constructor, without (); e.g. OlympicArena.
WORLD_KWARGS = {"load_precompiled": False}  # Constructor options for the selected world.
TERRAIN_SEED = 2026  # Fixed OlympicArena terrain; independent of the evolutionary seed.
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]  # Initial core X/Y/Z in metres.
SPAWN_ROTATION = [0, 0, 180]  # Initial roll/pitch/yaw in degrees; snake faces the target.
CORRECT_SPAWN_COLLISION = True  # Lift the body if it initially intersects the floor.
TARGET_POSITION: list[float] = [6.07, 0.0, 0.505]  # Target X/Y/Z; fitness uses all three coordinates.
TARGET_ON_FINAL_PLATFORM = True  # Derive OlympicArena target from the finish_end surface.
SIM_DURATION: float = 15.0  # Simulated seconds per candidate (--duration).
TARGET_RADIUS = 0.1  # Final 3D distance in metres labelled successful (--target-radius).
# Fitness version 2 uses XYZ distance to the surface target, plus terrain checks.
# TARGET_RADIUS now applies to 3D distance and requires physical terrain contact.
FALL_HEIGHT_MARGIN = 0.25  # Stop when the core drops this far below its spawn height.
FAILURE_PENALTY = 100.0  # Added to distance for falling or ending unsupported/off terrain.
MAX_CORE_CLEARANCE = 0.25  # Largest final core-to-terrain gap counted as on terrain (m).
TERRAIN_PENETRATION_TOLERANCE = 0.02  # Allow small numerical contact penetration (m).
BAD_FITNESS = 1_000_000.0  # Score assigned only to numerically invalid simulations.
SIMULATION_CHUNK_STEPS = 50  # Physics steps between validity checks/trajectory samples.

# --- Neural controller and initial weight distribution --- #
HIDDEN_SIZE: int = 8  # Hidden-layer neurons; changes the chromosome length.
INITIAL_WEIGHT_SIGMA = 0.5  # Standard deviation of zero-mean initial/random-search weights.
WEIGHT_LIMIT = 5.0  # Clip evolved and random-search weights to +/- this value.
CLOCK_FREQUENCIES = (0.75, 1.5)  # Clock inputs in Hz; each adds a sine and cosine input.
POSITION_INPUT_SCALE = np.pi  # Clip orientation/joint observations to +/- scale, then divide.
VELOCITY_INPUT_SCALE = 5.0  # Divide velocities by this before applying tanh.
TARGET_INPUT_SCALE = 2.0  # Divide the XY target gap by this many metres before tanh.
FORWARD_X_SIGN = -1.0  # Body forward is core-local -X for snake; use +1 for local +X.
ACTION_ANGLE_LIMIT = np.pi / 2  # Maximum magnitude of direct hinge commands in radians.

# --- Evolution and stopping rules (also available as command-line options) --- #
POPULATION_SIZE = 32  # Survivors retained AND new candidates per generation (--population).
GENERATIONS = 150  # Maximum offspring generations; additional on resume (--generations).
WORKERS = min(4, os.cpu_count() or 1)  # Parallel evaluation processes; 1 runs locally (--workers).
ALGORITHM = "ea"  # "ea" evolves parents; "random" samples independently (--algorithm).
SEARCH_MODE = "standard"  # "standard", "mixed", "focused" or experimental "adaptive" (--search-mode).
TOURNAMENT_SIZE = 3  # Random contestants per parent-selection tournament (--tournament).
CROSSOVER_RATE = 0.2  # Chance of crossover when the selected category permits (--crossover-rate).
MUTATION_RATE = 0.1  # Per-gene mutation chance, except focused fine edits (--mutation-rate).
MUTATION_SIGMA = 0.15  # Standard deviation of Gaussian mutation noise (--mutation-sigma).
MUTATION_OPERATOR = "gaussian"  # gaussian, uniform_step, or uniform_reset (--mutation-operator).
PATIENCE = 30  # Generations without significant improvement; 0 disables stopping (--patience).
MIN_IMPROVEMENT = 0.0001  # Improvement in metres required to reset patience (--min-improvement).
RUN_BASELINE = False  # Also run random search at the EA's actual evaluation budget (--baseline).
CROSSOVER_GENE_PROBABILITY = 0.5  # Chance of taking each gene from the second parent.
MUTATION_SCALES = (0.2, 1.0, 3.0)  # Noise multipliers sampled in mixed/focused modes.
MUTATION_SCALE_PROBABILITIES = (0.6, 0.3, 0.1)  # Matching category probabilities; must sum to 1.
FINE_MUTATION_THRESHOLD = 1.0  # Scales below this skip crossover and use focused single-gene edits.
ADAPTIVE_SIGMA_MIN = 0.002  # Smallest per-parent mutation sigma in adaptive search.
ADAPTIVE_SIGMA_MAX = 0.5  # Largest per-parent mutation sigma in adaptive search.
ADAPTIVE_SIGMA_TAU = 0.7  # Log-normal change strength for inherited mutation sigma.

# --- Files and workflow --- #
DATA = Path(__file__).resolve().parent  # Default results folder.
OUTPUT_DIR = None  # Optional custom training folder (--output-dir); None uses DATA.
REPLAY = None  # None trains; "latest" or a brain JSON path replays (--replay).
RESUME = None  # None starts fresh; "latest" or a saved run/checkpoint continues (--resume).
NO_VIEW = True  # Save videos without opening a window; --view explicitly opens one.
EXPORT_VIDEO = True  # Save best_replay.mp4 and its starting PNG; --no-video disables.
MODE: ViewerTypes = "launcher"  # Direct run_experiment() only: launcher/video/simple/frame/no_control.

# --- Target markers, camera, replay and exported figures --- #
TARGET_POLE_HEIGHT = 0.3  # Pole centre Z and half-height in metres (base at Z=0).
TARGET_POLE_RADIUS = 0.018  # Visual target pole radius in metres; never collides.
TARGET_POLE_COLOR = [1, 0.12, 0.12, 0.8]  # Pole RGBA colour.
TARGET_BALL_HEIGHT = 0.62  # Target ball centre Z in metres.
TARGET_BALL_RADIUS = 0.07  # Visual target ball radius in metres; never collides.
TARGET_BALL_COLOR = [1, 0.85, 0.05, 1]  # Ball RGBA colour.
# Marker heights above are offsets from TARGET_POSITION[2], the target surface.
CAMERA_LOOKAT = [0.65, 0, 0.1]  # Camera focus point in world coordinates.
CAMERA_DISTANCE = 5.5  # Camera distance from its focus point in metres.
CAMERA_AZIMUTH = 90  # Horizontal camera angle in degrees.
CAMERA_ELEVATION = -42  # Vertical camera angle in degrees.
AUTO_FRAME_TARGET = True  # Fit both the spawn and target in the replay camera.
VIEWER_CHUNK_STEPS = 10  # Physics steps between interactive display updates.
REPLAY_HOLD_SECONDS = 2.0  # Hold the final pose this long in the viewer and exported video.
VIEWER_POLL_SECONDS = 0.02  # Delay between window updates while holding the final pose.
VIDEO_FPS = 25  # Exported video frames per second.
VIDEO_SIZE = (960, 640)  # Exported video (width, height) in pixels.
VIDEO_CODEC = "mp4v"  # Four-character OpenCV video codec.
VIDEO_TEXT_POSITIONS = ((18, 28), (18, 54))  # Pixel origins for the two video caption lines.
VIDEO_TEXT_SCALE = 0.6  # OpenCV caption font scale.
VIDEO_TEXT_COLORS = ((255, 255, 255), (255, 235, 80))  # Caption colours in OpenCV BGR order.
VIDEO_TEXT_THICKNESS = 1  # Caption stroke width in pixels.
PLOT_SIZE = (8, 4.5)  # Convergence figure size in inches.
PLOT_DPI = 160  # Resolution of the saved convergence PNG.
PLOT_SPREAD_ALPHA = 0.2  # Opacity of the standard-deviation band.
PLOT_GRID_ALPHA = 0.25  # Opacity of convergence-plot grid lines.

# --- Derived paths and runtime state (not experiment settings) --- #
RNG = np.random.default_rng(SEED)
set_seed(SEED)  # ARIEL has a separate package-level random generator.
DATA.mkdir(parents=True, exist_ok=True)
LATEST_BRAIN = DATA / "latest_best.json"
_WORKER_MODEL = None
_WORKER_DATA = None


# ============================================================================ #
#  1. THE BODY AND THE WORLD
# ============================================================================ #
class _SeededOlympicArena(OlympicArena):
    """Assignment-local terrain generator; the ARIEL source stays unchanged.

    OlympicArena constructs PerlinNoise() without a seed. Reproduce its existing
    heightmap formula with an explicit seed so independent EA runs share a task.
    """

    def _generate_heightmap(self):
        size = self.rugged_resolution
        noise = PerlinNoise(seed=TERRAIN_SEED).as_grid(
            size, size, scale=self.rugged_hillyness, normalize=False,
        )
        u, v = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="xy")
        distance = np.minimum.reduce([u, 1 - u, v, 1 - v])
        t = np.clip(distance / getattr(self, "edge_width", 0.1), 0.1, 1.0)
        return noise * t * t * (3 - 2 * t)


def build_world() -> BaseWorld:
    """Create the environment the robot lives in.

    YOU MAY CHANGE THIS. Options include: SimpleFlatWorld, RuggedTerrainWorld,
    CraterTerrainWorld, AmphitheatreTerrainWorld, OlympicArena, ...
    (SimpleTiltedWorld is not supported for this task.)

    Whatever you pick, keep it FIXED for all runs you compare against each
    other, and say in your report which one you used. A controller evolved on
    flat ground and one evolved on rugged terrain are not comparable numbers.
    """
    # Constructor options belong with WORLD_FACTORY in EDITABLE SETTINGS.
    factory = _SeededOlympicArena if WORLD_FACTORY is OlympicArena else WORLD_FACTORY
    world = factory(**WORLD_KWARGS)
    if isinstance(world, OlympicArena) and TARGET_ON_FINAL_PLATFORM:
        terrain = world.spec.compile()
        finish = terrain.geom("finish_end")
        # Local +Z identifies the top surface, including its actual thickness.
        rotation = np.empty(9)
        mj.mju_quat2Mat(rotation, finish.quat)
        TARGET_POSITION[:] = (finish.pos + rotation.reshape(3, 3) @
                              np.array([0.0, 0.0, finish.size[2]])).tolist()
    # These visual-only geometries cannot push the snake or alter its fitness.
    # The red pole and yellow ball mark the target's exact X/Y coordinates.
    world.spec.worldbody.add_geom(
        name="target_pole", type=mj.mjtGeom.mjGEOM_CYLINDER,
        pos=[TARGET_POSITION[0], TARGET_POSITION[1], TARGET_POSITION[2] + TARGET_POLE_HEIGHT],
        size=[TARGET_POLE_RADIUS, TARGET_POLE_HEIGHT, 0.0], rgba=TARGET_POLE_COLOR,
        contype=0, conaffinity=0,
    )
    world.spec.worldbody.add_geom(
        name="target_ball", type=mj.mjtGeom.mjGEOM_SPHERE,
        pos=[TARGET_POSITION[0], TARGET_POSITION[1], TARGET_POSITION[2] + TARGET_BALL_HEIGHT],
        size=[TARGET_BALL_RADIUS, 0, 0], rgba=TARGET_BALL_COLOR,
        contype=0, conaffinity=0,
    )
    return world


def build_robot() -> CoreModule:
    """Create the robot body.

    YOU MAY CHANGE THIS. Options include the prebuilt bodies in
    `ariel.body_phenotypes.robogen_lite.prebuilt_robots` (gecko, spider, ...).

    Two consequences of this choice, and they matter:
      * The body determines `model.nu` (the number of hinges you must send
        commands to) - that is the OUTPUT size of your controller.
      * The body determines the size of `data.qpos` - if you feed qpos to your
        network, that is (part of) your INPUT size.
    Change the body and your genotype length changes with it. Keep the body
    FIXED within an experiment.
    """
    return BODY_FACTORY()


# ============================================================================ #
#  2. THE CONTROLLER CONTRACT
# ============================================================================ #
#
# MuJoCo calls the controller every physics step with (model, data); its job
# is to write into data.ctrl.
#
#   INPUTS   : whatever you read from `data` (qpos, qvel, time, ...), plus any
#              task info you already know, e.g. the vector to TARGET_POSITION.
#              INPUT SIZE is your choice, but must stay CONSTANT.
#   OUTPUTS  : exactly `model.nu` values, one per actuated hinge.
#   RANGE    : hinges accept [-pi/2, +pi/2] radians. A tanh output gives
#              [-1, 1] - rescale: actions * (np.pi / 2).
#   WRITING  : DIRECT (data.ctrl[:] = actions) commands the angle straight -
#              fast, but can destabilise the sim on large jumps. DELTA
#              (data.ctrl[:] += actions * alpha, alpha ~ 0.05, then clip) is
#              smoother but accumulates, so clipping is required. Pick one,
#              justify it, use it everywhere.
#   NaN      : blown-up weights silently write NaN into data.ctrl. Assert
#              against it while developing.
#
# ============================================================================ #

# IMPLEMENTATION NOTE: the original template comments/docstrings are retained
# as requested; references to a random-only demo describe the starting template.
# main() evolves a brain and saves its replay video. The John Set body is fixed.
# Clock inputs provide time information to the MLP, not a CPG controller or a
# prescribed gait: every mapping from observations to joint actions is evolved.


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    weights: list[npt.NDArray[np.float64]],
    duration: float = SIM_DURATION,
) -> npt.NDArray[np.float64]:
    """Map robot state to hinge commands: in -> hidden -> actions.

    In this demo `weights` is drawn at RANDOM. In your assignment, `weights`
    is what the evolutionary algorithm produces: an individual's genotype,
    reshaped into these matrices. You are free to change the architecture
    itself (layers, activations, ...) - just keep input/output sizes correct.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model. Use `model.nu` for the number of hinges.
    data : mj.MjData
        The MuJoCo data. This is where you read the robot's state from.
    weights : list of ndarray
        [w1, w2] - the layer weight matrices.
    duration : float
        Evaluation length in seconds, used to normalise elapsed time.

    Returns
    -------
    npt.NDArray[np.float64]
        `model.nu` action values, already scaled to [-pi/2, pi/2].
    """
    w1, w2 = weights

    # --- INPUTS ---------------------------------------------------------- #
    # Bare qpos - the simplest choice, not necessarily a good one. See
    # YOUR JOB below.
    # Use bounded state, target displacement, clock features and a bias input.
    # This replaces the bare-qpos demo while retaining its explanatory comments.
    inputs = _controller_inputs(data, duration)

    # --- FORWARD PASS ----------------------------------------------------- #
    layer1 = np.tanh(inputs @ w1)
    outputs = np.tanh(np.append(layer1, 1.0) @ w2)  # in [-1, 1]

    # --- RESCALE TO THE HINGE RANGE --------------------------------------- #
    return outputs * ACTION_ANGLE_LIMIT  # Direct desired angles in radians.


def make_random_weights(
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Draw a random parameter set for `nn_controller`.

    THIS IS THE FUNCTION YOUR EA REPLACES. Instead of sampling weights from a
    normal distribution, your EA will search for them.

    Note the total parameter count printed by main(): that is the length of the
    flat vector an individual's genotype has to encode. Reshaping a flat
    genotype back into these matrices is on you.
    """
    return [
        RNG.normal(scale=INITIAL_WEIGHT_SIGMA, size=(input_size, HIDDEN_SIZE)),
        RNG.normal(scale=INITIAL_WEIGHT_SIGMA, size=(HIDDEN_SIZE + 1, output_size)),
    ]


# ============================================================================ #
#  3. POSITION AND FITNESS
# ============================================================================ #
#
# The robot is spawned with a free joint, so data.qpos[0:3] IS the core's
# (x, y, z) world position. Read it before and after stepping - no tracker or
# bookkeeping needed. (`data.geom("robot1_core").xpos` works too.)
#
# ============================================================================ #


def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the robot core's current (x, y, z) world position."""
    return np.asarray(data.qpos[0:3]).copy()


def fitness_function(
    initial_position: npt.NDArray[np.float64],
    final_position: npt.NDArray[np.float64],
) -> float:
    """Score one evaluation. LOWER IS BETTER.

    The plain version: how far is the robot from the target when time runs out?

    `initial_position` is unused here on purpose - it is passed in because the
    moment you want a less naive fitness you will need it. Some things worth
    thinking about (and, ideally, comparing in your report):
      * Distance *reduced* rather than distance remaining, so a robot that
        starts closer is not rewarded for standing still.
      * Penalising a robot that falls over or leaves the arena.
      * Whether the z-axis should count at all - a robot that jumps is not
        closer to the target in any way you care about.
    See `ariel.simulation.tasks.targeted_locomotion` for some worked variants.
    """
    target = np.asarray(TARGET_POSITION)
    return float(np.linalg.norm(np.asarray(final_position) - target))


def _fitness_settings():
    """Version the objective separately from the unchanged neural-network layout."""
    return {"version": 2, "distance": "XYZ core to target surface",
            "fall_height_margin": FALL_HEIGHT_MARGIN, "failure_penalty": FAILURE_PENALTY,
            "max_core_clearance": MAX_CORE_CLEARANCE,
            "penetration_tolerance": TERRAIN_PENETRATION_TOLERANCE,
            "fall_check_steps": SIMULATION_CHUNK_STEPS, "success_requires_contact": True}


def _require_current_fitness(saved):
    """Never select survivors using scores produced by a different objective."""
    if saved.get("fitness_settings") != _fitness_settings():
        raise ValueError("This checkpoint uses different/legacy fitness. Start a fresh run without --resume. "
                         "The old archive remains available for diagnostic replay.")


def _fell_off(model, data):
    """A core well below the spawn surface has fallen out of this arena."""
    return bool(data.qpos[2] < model.qpos0[2] - FALL_HEIGHT_MARGIN)


def _task_status(model, data):
    """Measure real terrain support; visual markers and the robot cannot be ground."""
    position = get_core_position(data)
    distance_3d = fitness_function(None, position)
    distance_xy = float(np.linalg.norm(position[:2] - TARGET_POSITION[:2]))
    status = {"distance_3d": float(distance_3d) if np.isfinite(distance_3d) else None,
              "distance_xy": float(distance_xy) if np.isfinite(distance_xy) else None,
              "core_height": float(position[2]) if np.isfinite(position[2]) else None,
              "fallen": _fell_off(model, data),
              "on_terrain": False, "terrain_contact": False,
              "terrain_height": None, "core_clearance": None, "terrain_geom": None,
              "elapsed_seconds": float(data.time) if np.isfinite(data.time) else None}
    # An invalid simulation can leave non-finite state. Record unavailable
    # measurements as JSON null and do not feed that state back into MuJoCo.
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        return status
    # Synchronise contacts and geometry with qpos after the last integration step.
    mj.mj_forward(model, data)
    terrain = ((model.body_weldid[model.geom_bodyid] == 0)
               & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
    status["terrain_contact"] = any(
        terrain[contact.geom1] != terrain[contact.geom2] and contact.dist <= 0.001
        for contact in data.contact
    )
    origin = position.copy()
    origin[2] = max(position[2], TARGET_POSITION[2],
                    model.stat.center[2] + model.stat.extent) + 1.0
    hit = np.array([-1], dtype=np.int32)
    groups = model.geom_group.copy()
    try:
        # Ray masks use visual groups. Restore them immediately: no physics or
        # archived geometry changes, and no robot self-intersection in the ray.
        model.geom_group[:] = np.where(terrain, 0, 1)
        distance = mj.mj_ray(model, data, origin, np.array([0.0, 0.0, -1.0]),
                             np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), 1, -1, hit)
    finally:
        model.geom_group[:] = groups
    if distance >= 0:
        height = float(origin[2] - distance)
        clearance = float(position[2] - height)
        status.update(terrain_height=height, core_clearance=clearance,
                      terrain_geom=model.geom(int(hit[0])).name,
                      on_terrain=bool(not status["fallen"] and
                                      -TERRAIN_PENETRATION_TOLERANCE <= clearance <= MAX_CORE_CLEARANCE))
    return status


def _score_state(model, data):
    """Keep geometric distance separate from the penalty used for selection."""
    status = _task_status(model, data)
    score = status["distance_3d"]
    if score is None:
        return BAD_FITNESS, status
    if status["fallen"] or not status["on_terrain"]:
        score += FAILURE_PENALTY
    return float(score), status


def _format_distance(value, precision=6):
    """Keep unavailable measurements readable without inventing a distance."""
    return "unavailable" if value is None else f"{value:.{precision}f} m"


def _reached_target(score, valid, status, radius):
    """Being horizontally above/below the goal or airborne cannot count as success."""
    return bool(valid and not status["fallen"] and status["on_terrain"]
                and status["terrain_contact"] and score <= radius)


def _advance_with_fall_check(model, data, steps):
    """Give viewer/video frames the same fall-check boundaries as evaluations."""
    while steps > 0:
        current = int(round(data.time / model.opt.timestep))
        chunk = min(steps, SIMULATION_CHUNK_STEPS - current % SIMULATION_CHUNK_STEPS)
        mj.mj_step(model, data, nstep=chunk)
        steps -= chunk
        if (current + chunk) % SIMULATION_CHUNK_STEPS == 0 and _fell_off(model, data):
            return False
    return True



# ============================================================================ #
#  4. RUNNING ONE EVALUATION
# ============================================================================ #


def run_experiment(
    mode: ViewerTypes = MODE,
    weights: list[npt.NDArray[np.float64]] | None = None,
) -> float:
    """Set up the world, run one simulation, and return the fitness.

    This is the function your EA calls once per individual, with `mode` set
    to "simple" (headless).

    Returns
    -------
    float
        The fitness of this run. Lower is better.
    """
    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    # --- World and robot --------------------------------------------------- #
    world = build_world()
    robot = build_robot()

    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        rotation=SPAWN_ROTATION,
        correct_collision_with_floor=CORRECT_SPAWN_COLLISION,
    )

    # Compile the world into a model. USE AS IS.
    model = world.spec.compile()
    data = mj.MjData(model)

    # Put the simulation in a clean, known state before reading anything.
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # --- Wire up the controller -------------------------------------------- #
    # Sizes are read from the compiled model, never hardcoded - they depend on
    # the body you chose in build_robot().
    input_size = len(_controller_inputs(data))
    output_size = model.nu

    duration = SIM_DURATION
    if weights is None:
        # Direct calls replay the most recently evolved brain when available.
        if LATEST_BRAIN.exists():
            brain_path = Path(json.loads(LATEST_BRAIN.read_text())["brain"])
            saved = json.loads(brain_path.read_text(encoding="utf-8"))
            # The direct Python entry point must use the same archived physics
            # as --replay, even after build_world() has been edited.
            model = _archived_model(brain_path, saved)
            data = mj.MjData(model)
            mj.mj_resetData(model, data)
            mj.mj_forward(model, data)
            weights = _load_brain(brain_path, model)
            duration = saved["duration"]
        else:
            weights = make_random_weights(input_size, output_size)

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        """Compute and apply actions; MuJoCo calls this every physics step."""
        actions = nn_controller(m, d, weights, duration)
        if not np.all(np.isfinite(actions)):
            raise FloatingPointError("The controller produced non-finite actions.")

        # DIRECT application (see the controller contract above).
        d.ctrl[:] = actions

        # DELTA application - comment out the line above and use these instead:
        # delta = 0.05
        # d.ctrl[:] += actions * delta
        # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    # --- Record the starting point ----------------------------------------- #
    initial_position = get_core_position(data)

    # --- Run ---------------------------------------------------------------- #
    if mode != "no_control":
        mj.set_mjcb_control(control_callback)

    try:
        match mode:
            case "launcher":
                # Interactive window. Great for seeing what your robot does,
                # useless inside an evolutionary loop.
                _interactive_replay(model, data, weights, duration)
            case "simple":
                # Headless. THIS is the one your EA uses.
                _, _, valid = _simulate(model, data, weights, duration)
                if not valid:
                    return BAD_FITNESS
            case "video":
                # Render to an mp4 - for the figures in your report.
                _export_video(model, weights, duration, DATA / "replay.mp4")
                _, _, valid = _simulate(model, data, weights, duration)
                if not valid:
                    return BAD_FITNESS
            case "frame":
                # A single image of the scene. Useful to check your spawn position
                # and that the robot is not clipping through the floor.
                single_frame_renderer(model, data, steps=1, show=True)
            case "no_control":
                # No controller attached: drag the hinges around by hand.
                viewer.launch(model=model, data=data)
            case _:
                raise ValueError(f"Unknown viewer mode: {mode}")
    finally:
        # Detach the callback again so the next run starts clean.
        mj.set_mjcb_control(None)

    # --- Score -------------------------------------------------------------- #
    final_position = get_core_position(data)
    fitness, status = _score_state(model, data)

    console.log(f"start  : {np.round(initial_position, 3)}")
    console.log(f"end    : {np.round(final_position, 3)}")
    console.log(f"target : {np.round(TARGET_POSITION, 3)}")
    console.log(f"fitness: {fitness:.4f}   (lower is better); 3D distance: {_format_distance(status['distance_3d'], 4)}; "
                f"fallen: {status['fallen']}; on terrain: {status['on_terrain']}")

    return fitness


def main() -> None:
    """Inspect controller dimensions and dispatch training, resume or replay."""
    if "--mutation-study" in sys.argv:
        from mutation_study_A2_2026 import _main
        _main([value for value in sys.argv[1:] if value != "--mutation-study"])
        return
    # A quick look at the size of the problem you are about to search.
    mj.set_mjcb_control(None)
    world = build_world()
    robot = build_robot()
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        rotation=SPAWN_ROTATION,
        correct_collision_with_floor=CORRECT_SPAWN_COLLISION,
    )
    model = world.spec.compile()
    data = mj.MjData(model)

    input_size = len(_controller_inputs(data))
    output_size = model.nu
    num_weights = (
        input_size * HIDDEN_SIZE
        + (HIDDEN_SIZE + 1) * output_size
    )
    console.log(f"controller inputs (state/target/time/bias): {input_size}")
    console.log(f"controller outputs (model.nu)      : {output_size}")
    console.log(f"genotype length (total weights)    : {num_weights}")

    _cli(model)


def _heading_features(data: mj.MjData) -> tuple[float, float]:
    """Return sine/cosine of the signed planar turn from forward to target."""
    # This snake's tail attaches along core-local +X, so its forward direction
    # is local -X. MuJoCo stores the root quaternion in w, x, y, z order.
    # With our 180-degree spawn yaw, forward is world +X, toward the target.
    w, x, y, z = data.qpos[3:7]
    forward_x = FORWARD_X_SIGN * (1.0 - 2.0 * (y * y + z * z))
    forward_y = FORWARD_X_SIGN * 2.0 * (x * y + w * z)
    target_x = TARGET_POSITION[0] - data.qpos[0]
    target_y = TARGET_POSITION[1] - data.qpos[1]
    scale = np.hypot(forward_x, forward_y) * np.hypot(target_x, target_y)
    # At the target, or with a vertical forward axis, planar heading is
    # undefined. Supply a finite neutral direction instead of dividing by zero.
    if scale < 1e-12:
        return 0.0, 1.0
    sine = (forward_x * target_y - forward_y * target_x) / scale
    cosine = (forward_x * target_x + forward_y * target_y) / scale
    # Sine/cosine avoid an abrupt angle jump between -pi and +pi.
    return float(np.clip(sine, -1.0, 1.0)), float(np.clip(cosine, -1.0, 1.0))


def _input_size(model: mj.MjModel) -> int:
    """State + target XY + clock pairs + time + heading pair + bias."""
    return model.nq - 3 + model.nv + 2 * len(CLOCK_FREQUENCIES) + 6


def _controller_inputs(data: mj.MjData, duration: float = SIM_DURATION) -> np.ndarray:
    """Bounded proprioception + world-frame target displacement + clock + bias.

    qpos[3:] contains core orientation and hinge angles; qvel contains the
    floating-base and hinge velocities. Two fixed clock frequencies make
    rhythmic actions representable without encoding a hand-designed snake gait.
    """
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Controller duration must be finite and positive.")
    phases = 2 * np.pi * data.time * np.asarray(CLOCK_FREQUENCIES)
    # Unlike the repeating clocks, elapsed time tells the brain how much of
    # this evaluation has passed. Training and replay pass the same duration.
    elapsed = np.clip(data.time / duration, 0.0, 1.0)
    return np.concatenate([
        np.clip(data.qpos[3:], -POSITION_INPUT_SCALE, POSITION_INPUT_SCALE) / POSITION_INPUT_SCALE,
        np.tanh(data.qvel / VELOCITY_INPUT_SCALE),
        np.tanh((np.asarray(TARGET_POSITION[:2]) - data.qpos[:2]) / TARGET_INPUT_SCALE),
        np.sin(phases), np.cos(phases), [elapsed], _heading_features(data), [1.0],
    ])


def _decode(genotype, model: mj.MjModel) -> list[np.ndarray]:
    """Decode all NN weights/biases from one real-valued chromosome."""
    inputs = _input_size(model)
    split = inputs * HIDDEN_SIZE
    genes = np.asarray(genotype, dtype=np.float64)
    expected = split + (HIDDEN_SIZE + 1) * model.nu
    if genes.shape != (expected,) or not np.all(np.isfinite(genes)):
        raise ValueError(f"Expected {expected} finite genes; got {genes.shape}.")
    return [genes[:split].reshape(inputs, HIDDEN_SIZE),
            genes[split:].reshape(HIDDEN_SIZE + 1, model.nu)]


def _simulate(model, data, weights, duration, *, trace=False):
    """Reset every state and evaluate exactly the requested number of steps.

    The callback is per-process, never shared by threads. try/finally prevents
    an exception from leaving a stale controller attached to the next robot.
    Physics and control both run at the stock 500 Hz in training and replay.
    """
    mj.set_mjcb_control(None)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    initial = get_core_position(data)
    trajectory = [[0.0, *initial]]

    def _control_callback(m, d):
        actions = nn_controller(m, d, weights, duration)
        if not np.all(np.isfinite(actions)):
            raise FloatingPointError("Non-finite neural-network output")
        d.ctrl[:] = actions

    total = int(round(duration / model.opt.timestep))
    try:
        mj.set_mjcb_control(_control_callback)
        for start in range(0, total, SIMULATION_CHUNK_STEPS):
            mj.mj_step(model, data, nstep=min(SIMULATION_CHUNK_STEPS, total - start))
            # MuJoCo may auto-reset after an unstable step, so check warnings
            # as well as finite coordinates and the simulated clock.
            if (not np.all(np.isfinite(data.qpos))
                    or not np.all(np.isfinite(data.qvel))
                    or any(data.warning[k].number for k in (
                        mj.mjtWarning.mjWARN_BADQPOS,
                        mj.mjtWarning.mjWARN_BADQVEL,
                        mj.mjtWarning.mjWARN_BADQACC))):
                return BAD_FITNESS, trajectory, False
            if trace:
                trajectory.append([float(data.time), *get_core_position(data)])
            if _fell_off(model, data):
                # A physical fall is a failed task, not a numerical solver error.
                return _score_state(model, data)[0], trajectory, True
        if not np.isclose(data.time, total * model.opt.timestep, atol=1e-8):
            return BAD_FITNESS, trajectory, False
        return _score_state(model, data)[0], trajectory, True
    except FloatingPointError:
        return BAD_FITNESS, trajectory, False
    finally:
        mj.set_mjcb_control(None)


def _worker_init(model_path, target=None):
    """Load the exact saved model once in each Windows-spawned worker."""
    global _WORKER_MODEL, _WORKER_DATA
    if target is not None:
        TARGET_POSITION[:] = target
    mj.set_mjcb_control(None)
    _WORKER_MODEL = mj.MjModel.from_binary_path(model_path)
    _WORKER_DATA = mj.MjData(_WORKER_MODEL)


def _worker_evaluate(job):
    genes, duration = job
    score, _, valid = _simulate(
        _WORKER_MODEL, _WORKER_DATA, _decode(genes, _WORKER_MODEL), duration,
    )
    final = get_core_position(_WORKER_DATA)
    # Preserve the failure without putting NaN/Infinity into SQLite or JSON.
    final = [float(value) if np.isfinite(value) else None for value in final]
    return score, final, valid, _task_status(_WORKER_MODEL, _WORKER_DATA)


def _new_individual(genes) -> Individual:
    individual = Individual()
    individual.genotype = np.asarray(genes, dtype=float).tolist()
    return individual


def _evaluate_population(population: Population, pool, duration) -> Population:
    pending = list(population.unevaluated)
    jobs = [(individual.genotype, duration) for individual in pending]
    results = pool.map(_worker_evaluate, jobs) if pool else map(_worker_evaluate, jobs)
    for individual, (score, final, valid, status) in zip(pending, results, strict=True):
        individual.fitness = score
        # Adaptive offspring carry their own mutation sigma into selection and
        # checkpoints. Evaluation adds measurements without erasing that state.
        individual.tags = {**individual.tags, **status, "final_position": final, "valid": valid}
    return population


def _mutate(genes, mask, rng, operator, sigma):
    """Real-valued mutation; mask/rate and clipping are shared across operators.

    Uniform reset is textbook section 4.4.1: replace selected genes in [-L,L].
    Uniform step is an additional local comparison with the same pre-clipping
    standard deviation as Gaussian noise (half-width sqrt(3)*sigma).
    Empty masks are allowed: forcing one edit would change the declared rate.
    """
    count = int(mask.sum())
    if operator == "gaussian":
        genes[mask] += rng.normal(0, sigma, count)
    elif operator == "uniform_step":
        width = np.sqrt(3.0) * sigma
        genes[mask] += rng.uniform(-width, width, count)
    elif operator == "uniform_reset":
        genes[mask] = rng.uniform(-WEIGHT_LIMIT, WEIGHT_LIMIT, count)
    else:
        raise ValueError(f"Unknown mutation operator: {operator}")
    return np.clip(genes, -WEIGHT_LIMIT, WEIGHT_LIMIT)


def _breed(population: Population, rng, args, length) -> Population:
    """Tournament parents, optional uniform crossover, then real-valued mutation.

    Each generation evaluates exactly population-size NEW candidates. Parents
    are not reevaluated because the task is deterministic. New Individual
    objects prevent stale fitness values from following a changed chromosome.
    """
    parents = list(population.alive)
    for _ in range(args.population):
        child_sigma = None
        if args.algorithm == "random":
            genes = np.clip(rng.normal(0, INITIAL_WEIGHT_SIGMA, length), -WEIGHT_LIMIT, WEIGHT_LIMIT)
        elif getattr(args, "search_mode", "standard") == "adaptive":
            # Self-adaptation: selection rewards both a useful brain and the
            # mutation strength that produced it. No crossover breaks up the
            # coordinated network. The sigma is strategy state, not a NN gene.
            contestants = rng.integers(0, len(parents), size=args.tournament)
            parent = min((parents[i] for i in contestants), key=lambda ind: ind.fitness)
            parent_sigma = parent.tags.get("mutation_sigma", args.mutation_sigma)
            child_sigma = float(np.clip(parent_sigma * np.exp(ADAPTIVE_SIGMA_TAU * rng.normal()),
                                        ADAPTIVE_SIGMA_MIN, ADAPTIVE_SIGMA_MAX))
            genes = np.asarray(parent.genotype).copy()
            mask = rng.random(length) < args.mutation_rate
            if not mask.any():
                mask[rng.integers(length)] = True
            genes[mask] += rng.normal(0, child_sigma, int(mask.sum()))
            genes = np.clip(genes, -WEIGHT_LIMIT, WEIGHT_LIMIT)
            if np.array_equal(genes, parent.genotype):
                index = rng.integers(length)
                direction = -1.0 if genes[index] >= 0 else 1.0
                genes[index] = np.clip(genes[index] + direction * child_sigma,
                                       -WEIGHT_LIMIT, WEIGHT_LIMIT)
        else:
            # Mixed scales spend most evaluations refining coordinated motion,
            # while retaining occasional larger changes to explore other gaits.
            # Standard mode preserves the original operator and RNG sequence.
            mode = getattr(args, "search_mode", "standard")
            scale = (float(rng.choice(MUTATION_SCALES, p=MUTATION_SCALE_PROBABILITIES))
                     if mode in ("mixed", "focused") else 1.0)
            selected = []
            for _ in range(2):
                contestants = rng.integers(0, len(parents), size=args.tournament)
                selected.append(min((parents[i] for i in contestants),
                                    key=lambda individual: individual.fitness))
            genes = np.asarray(selected[0].genotype).copy()
            if rng.random() < args.crossover_rate and scale >= FINE_MUTATION_THRESHOLD:
                mask = rng.random(length) < CROSSOVER_GENE_PROBABILITY
                genes[mask] = np.asarray(selected[1].genotype)[mask]
            if mode == "focused" and scale < FINE_MUTATION_THRESHOLD:
                # A good gait depends on coordinated weights. Fine offspring
                # change just ONE weight, without recombining the network.
                # The other 40% retain broader exploration of the search space.
                mask = np.zeros(length, dtype=bool)
                mask[rng.integers(length)] = True
            else:
                mask = rng.random(length) < args.mutation_rate
            genes = _mutate(genes, mask, rng, getattr(args, "mutation_operator", "gaussian"),
                            args.mutation_sigma * scale)
            if mode == "focused" and np.array_equal(genes, selected[0].genotype):
                # An empty mask or clipping at the boundary can undo mutation.
                # Spend this evaluation on an actual change, pointing inward
                # if the selected weight is already at its allowed limit.
                index = rng.integers(length)
                delta = max(abs(rng.normal(0, args.mutation_sigma * scale)), 1e-12)
                direction = -1.0 if genes[index] >= 0 else 1.0
                genes[index] = np.clip(genes[index] + direction * delta,
                                       -WEIGHT_LIMIT, WEIGHT_LIMIT)
        child = _new_individual(genes)
        if child_sigma is not None:
            child.tags = {"mutation_sigma": child_sigma}
        population.append(child)
    return population


def _survive(population: Population, size, unique=False) -> Population:
    """Elitist (mu + lambda) selection; lower distance always wins.

    Retain rejected candidates until ARIEL commits this generation so the
    SQLite archive contains every evaluated chromosome, including failures.
    """
    ranked = sorted(population.alive, key=lambda individual: individual.fitness)
    if unique:
        # Keep the best copy of each chromosome before filling spare slots
        # with duplicates. Fitness remains the only ranking among unique brains,
        # and the best-so-far controller is never lost.
        seen, distinct, duplicates = set(), [], []
        for individual in ranked:
            key = tuple(individual.genotype)
            if key in seen:
                duplicates.append(individual)
            else:
                seen.add(key)
                distinct.append(individual)
        ranked = distinct + duplicates
    for individual in ranked[size:]:
        individual.alive = False
    return population


def _atomic_json(path, payload):
    """Replace a complete JSON document atomically, never a half-written brain."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _body_name():
    """Derive the archive label from the selected constructor."""
    return f"{BODY_FACTORY.__module__.rsplit('.', 1)[-1]}.{BODY_FACTORY.__name__}"


def _controller_settings():
    """Settings that affect the meaning of a saved chromosome, not just its size."""
    return {"clock_frequencies": list(CLOCK_FREQUENCIES),
            "position_input_scale": POSITION_INPUT_SCALE,
            "velocity_input_scale": VELOCITY_INPUT_SCALE,
            "target_input_scale": TARGET_INPUT_SCALE,
            "forward_x_sign": FORWARD_X_SIGN, "action_angle_limit": ACTION_ANGLE_LIMIT}


def _metadata(model, args, seed, model_path):
    return {
        "schema": 3, "body": _body_name(), "world": _world_name(model),
        "terrain_seed": (TERRAIN_SEED if WORLD_FACTORY is OlympicArena
                         and not WORLD_KWARGS.get("load_precompiled", True) else None),
        "terrain_generation": ("seeded_perlin" if WORLD_FACTORY is OlympicArena
                               and not WORLD_KWARGS.get("load_precompiled", True) else "factory_defined"),
        "spawn": SPAWN_POS,
        "rotation_degrees": SPAWN_ROTATION, "target": list(TARGET_POSITION),
        "duration": args.duration, "hidden_size": HIDDEN_SIZE,
        "input_size": _input_size(model), "output_size": model.nu,
        "controller": "tanh MLP with biases, state/target/clock/elapsed time/heading sine+cosine",
        "controller_settings": _controller_settings(),
        "variation_settings": {"initial_weight_sigma": INITIAL_WEIGHT_SIGMA,
            "weight_limit": WEIGHT_LIMIT, "crossover_gene_probability": CROSSOVER_GENE_PROBABILITY,
            "mutation_scales": list(MUTATION_SCALES),
            "mutation_scale_probabilities": list(MUTATION_SCALE_PROBABILITIES),
            "fine_mutation_threshold": FINE_MUTATION_THRESHOLD,
            "adaptive_sigma_min": ADAPTIVE_SIGMA_MIN, "adaptive_sigma_max": ADAPTIVE_SIGMA_MAX,
            "adaptive_sigma_tau": ADAPTIVE_SIGMA_TAU},
        "control": "direct angle commands every physics step",
        "fitness": "3D core-to-target-surface distance plus fall/off-terrain penalty, minimised",
        "fitness_settings": _fitness_settings(),
        "timestep": model.opt.timestep, "seed": seed,
        "mujoco_version": mj.__version__, "numpy_version": np.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "settings": {key: value for key, value in vars(args).items()
                     if not key.startswith("_") and isinstance(value, (str, float, int, bool, list, type(None)))},
    }


def _world_name(model):
    """Read the actual compiled world, rather than a hardcoded experiment label."""
    return model.names.split(b"\0", 1)[0].decode("utf-8")


def _resolve_brain(value, output_root=None):
    """Resolve a brain, seed directory or numbered batch, including old latest pointers."""
    if value == "latest":
        pointer = Path(output_root) / "latest_best.json" if output_root else LATEST_BRAIN
        if not pointer.exists() and output_root is None:
            pointer = DATA / "__data__" / Path(__file__).stem / "latest_best.json"
        return Path(json.loads(pointer.read_text(encoding="utf-8"))["brain"])
    path = Path(value).resolve()
    if path.is_dir():
        if (path / "best_brain.json").exists():
            return path / "best_brain.json"
        chosen = json.loads((path / "latest_best.json").read_text(encoding="utf-8"))
        # Use the local copy when an entire archive has been moved.
        return path / Path(chosen["brain"]).parent.name / "best_brain.json"
    return path


def _new_run_directory(output_root, model):
    """Reserve the next body_world_number directory without overwriting a run."""
    output_root.mkdir(parents=True, exist_ok=True)
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{BODY_FACTORY.__name__}_{_world_name(model)}").lower()
    numbers = [int(match.group(1)) for path in output_root.iterdir()
               if (match := re.fullmatch(re.escape(label) + r"_(\d+)", path.name))]
    number = max(numbers, default=0) + 1
    while True:
        path = output_root / f"{label}_{number}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            number += 1


def _restore_target(saved, model):
    """Replay the archived task even when today's target defaults have changed."""
    target = np.asarray(saved["target"], dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Invalid archived target")
    if not np.allclose(model.geom("target_ball").pos[:2], target[:2], atol=1e-12, rtol=0):
        raise ValueError("Archived target disagrees with the saved model marker")
    TARGET_POSITION[:] = target.tolist()


def _resolve_run(value, output_root=None):
    """Accept 'latest', a run directory, a brain, or a checkpoint path."""
    path = Path(value).resolve() if value != "latest" else _resolve_brain(value, output_root)
    if path.is_dir() and (path / "latest_best.json").exists():
        path = _resolve_brain(str(path))
    return path if path.is_dir() else path.parent


def _continuation_state(run):
    """Prefer an atomic checkpoint; migrate older runs from their SQLite archive.

    Old runs did not save RNG state. Their surviving population is recoverable,
    but their future random sequence is not: that is a seeded warm start, not
    an exact resume. New checkpoints preserve both, at generation boundaries.
    """
    checkpoint = run / "checkpoint.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("checkpoint_schema") != 1:
            raise ValueError("Unsupported checkpoint version")
        return state
    brain = json.loads((run / "best_brain.json").read_text(encoding="utf-8"))
    # Read-only mode prevents an incorrect path from creating an empty database.
    with sqlite3.connect((run / "evolution.sqlite").resolve().as_uri() + "?mode=ro",
                         uri=True) as connection:
        rows = connection.execute(
            "SELECT genotype_, fitness_, tags_ FROM individual WHERE alive=1 ORDER BY id"
        ).fetchall()
        maximum_generation = connection.execute(
            "SELECT MAX(time_of_death) FROM individual"
        ).fetchone()[0]
    connection.close()
    if maximum_generation != brain["generation"]:
        raise ValueError("Legacy database and saved brain refer to different generations")
    return {"checkpoint_schema": 1, "metadata": brain,
            "generation": brain["generation"], "evaluations": brain["evaluations"],
            "rng_state": None, "previous_best": brain["fitness"],
            "last_improvement": brain["generation"],
            "population": [{"genotype": json.loads(genes), "fitness": score,
                            "tags": json.loads(tags)} for genes, score, tags in rows]}


def _save_checkpoint(ea, rng, metadata, evaluations, previous_best,
                     last_improvement, run_dir):
    """Atomically store the population and RNG after a completed generation."""
    state = {"checkpoint_schema": 1, "metadata": metadata,
             "generation": ea.current_generation, "evaluations": evaluations,
             "previous_best": previous_best, "last_improvement": last_improvement,
             "rng_state": rng.bit_generator.state,
             "population": [{"genotype": ind.genotype, "fitness": ind.fitness,
                             "tags": ind.tags} for ind in ea.population.alive]}
    _atomic_json(run_dir / "checkpoint.json", state)


def _record_generation(ea, generation, evaluations, run_dir, metadata, history):
    # ARIEL's commit expires SQLModel objects. Fetch fresh rows before reading
    # fields after a commit; accessing the detached objects raises an ORM error.
    ea.fetch_population()
    alive = list(ea.population.alive)
    scores = [individual.fitness for individual in alive]
    best = min(alive, key=lambda individual: individual.fitness)
    row = {"generation": generation, "evaluations": evaluations,
           "best": min(scores), "mean": float(np.mean(scores)), "worst": max(scores)}
    row["fitness_std"] = float(np.std(scores))
    row["valid_fraction"] = float(np.mean([ind.tags["valid"] for ind in alive]))
    row["fallen_fraction"] = float(np.mean([ind.tags["fallen"] for ind in alive]))
    row["on_terrain_fraction"] = float(np.mean([ind.tags["on_terrain"] for ind in alive]))
    row["best_distance_3d"] = best.tags["distance_3d"]
    row["best_core_height"] = best.tags["core_height"]
    row["target_fraction"] = float(np.mean([
        _reached_target(ind.fitness, ind.tags["valid"], ind.tags, metadata["settings"]["target_radius"])
        for ind in alive
    ]))
    # Track whether the population still contains meaningfully different brains.
    # This is a diagnostic, not a fitness bonus or a change to selection.
    genes = np.asarray([individual.genotype for individual in alive])
    row["mean_gene_std"] = float(genes.std(axis=0).mean())
    row["unique_genotypes"] = int(len(np.unique(genes, axis=0)))
    if (metadata["settings"].get("search_mode") == "adaptive"
            and metadata["settings"]["algorithm"] == "ea"):
        row["mean_mutation_sigma"] = float(np.mean([
            ind.tags.get("mutation_sigma", metadata["settings"]["mutation_sigma"]) for ind in alive
        ]))
    first_row = not history
    history.append(row)
    with (run_dir / "history.csv").open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if first_row:
            writer.writeheader()
        writer.writerow(row)
    payload = {**metadata, "generation": generation, "evaluations": evaluations,
               "fitness": best.fitness, "genotype": best.genotype,
               "final_position": best.tags["final_position"],
               "valid": best.tags["valid"], "task_status": dict(best.tags)}
    _atomic_json(run_dir / "best_brain.json", payload)
    print(f"{metadata['settings']['algorithm']} seed={metadata['seed']} "
          f"generation={generation:3d} evaluations={evaluations:5d} "
          f"best_fitness={row['best']:.5f} mean_fitness={row['mean']:.5f} "
          f"best_3d={_format_distance(row['best_distance_3d'], 5)} "
          f"valid={row['valid_fraction']:.0%} fallen={row['fallen_fraction']:.0%}", flush=True)
    return best


def _train(model, args, seed, run_dir, pool, model_path):
    """Run the custom EA through ARIEL's operations and SQLite persistence."""
    rng = np.random.default_rng(seed)
    set_seed(seed)
    length = _input_size(model) * HIDDEN_SIZE + (HIDDEN_SIZE + 1) * model.nu
    started = time.perf_counter()
    start_generation, inherited_evaluations = 0, 0
    state = getattr(args, "_continuation", None)
    if state:
        _require_current_fitness(state["metadata"])
    adaptive_changed = bool(state and getattr(args, "search_mode", "standard") == "adaptive" and (
        state["metadata"]["settings"].get("search_mode") != "adaptive"
        or state["metadata"]["settings"]["mutation_sigma"] != args.mutation_sigma
        or any(state["metadata"].get("variation_settings", {}).get(key) != value for key, value in (
            ("adaptive_sigma_min", ADAPTIVE_SIGMA_MIN), ("adaptive_sigma_max", ADAPTIVE_SIGMA_MAX),
            ("adaptive_sigma_tau", ADAPTIVE_SIGMA_TAU)))
    ))
    if state is None:
        population = Population([
            _new_individual(np.clip(rng.normal(0, INITIAL_WEIGHT_SIGMA, length), -WEIGHT_LIMIT, WEIGHT_LIMIT))
            for _ in range(args.population)
        ])
        _evaluate_population(population, pool, args.duration)
        evaluations = args.population
    else:
        population = Population([])
        for record in state["population"]:
            _decode(record["genotype"], model)
            individual = _new_individual(record["genotype"])
            individual.fitness = record["fitness"]
            individual.tags = record["tags"]
            if adaptive_changed:
                # An explicit strategy change starts from the requested sigma;
                # unchanged adaptive resumes preserve each inherited sigma.
                individual.tags = {**individual.tags, "mutation_sigma": args.mutation_sigma}
            population.append(individual)
        if len(population) != args.population:
            raise ValueError("Continuation must keep the saved population size")
        start_generation = state["generation"]
        evaluations = inherited_evaluations = state["evaluations"]
        if state["rng_state"] is not None:
            rng.bit_generator.state = state["rng_state"]
    operations = [
        EAOperation(_breed, rng, args, length),
        EAOperation(_evaluate_population, pool, args.duration),
        EAOperation(_survive, args.population,
                    getattr(args, "search_mode", "standard") in ("focused", "adaptive")
                    and args.algorithm == "ea"),
    ]
    ea = EA(population, operations, num_steps=args.generations,
            is_maximisation=False, first_generation_id=start_generation, quiet=True,
            db_file_path=run_dir / "evolution.sqlite", db_handling="halt")
    metadata = _metadata(model, args, seed, model_path)
    if state:
        # Resume uses the archived model, even if today's world settings differ.
        # Old files recorded TERRAIN_SEED despite never applying it; do not
        # relabel that old geometry as the newly seeded arena.
        for key in ("body", "world", "spawn", "rotation_degrees", "target", "timestep"):
            if key in state["metadata"]:
                metadata[key] = state["metadata"][key]
        metadata["terrain_generation"] = state["metadata"].get("terrain_generation", "legacy_unknown")
        metadata["terrain_seed"] = (state["metadata"].get("terrain_seed")
                                    if metadata["terrain_generation"] == "seeded_perlin" else None)
    metadata["continuation"] = {
        "source": str(args.resume) if state else None,
        "start_generation": start_generation, "inherited_evaluations": inherited_evaluations,
        "restored_rng": bool(state and state["rng_state"] is not None),
    }
    _atomic_json(run_dir / "config.json", metadata)
    history = []
    best = _record_generation(ea, start_generation, evaluations, run_dir, metadata, history)
    last_improvement = state["last_improvement"] if state else 0
    previous_best = state["previous_best"] if state else best.fitness
    # An intentional operator change begins a new patience window, so a previous
    # plateau cannot prematurely terminate the new refinement experiment.
    if state and (adaptive_changed or any(state["metadata"]["settings"].get(key, "standard" if key == "search_mode" else None)
                     != getattr(args, key) for key in
                     ("search_mode", "mutation_rate", "mutation_sigma", "crossover_rate", "tournament"))
                  or state["metadata"]["settings"].get("mutation_operator", "gaussian")
                     != getattr(args, "mutation_operator", "gaussian")):
        last_improvement, previous_best = start_generation, best.fitness
    _save_checkpoint(ea, rng, metadata, evaluations, previous_best, last_improvement, run_dir)
    stop_reason = "generation budget"
    plotting_seconds = 0.0
    try:
        for generation in range(start_generation + 1, start_generation + args.generations + 1):
            # A tuner may impose a wall-clock limit. Pause only between complete
            # generations, with the population and random state already saved.
            if time.perf_counter() >= getattr(args, "_deadline", float("inf")):
                stop_reason = "wall clock budget"
                break
            ea.step()
            evaluations += args.population
            best = _record_generation(ea, generation, evaluations,
                                      run_dir, metadata, history)
            if best.fitness < previous_best - args.min_improvement:
                previous_best = best.fitness
                last_improvement = generation
            _save_checkpoint(ea, rng, metadata, evaluations, previous_best, last_improvement, run_dir)
            if args.patience and generation - last_improvement >= args.patience:
                stop_reason = "fitness plateau"
                break
    finally:
        ea.engine.dispose()
        # Also leave diagnostics for completed generations after an interruption.
        plotting_started = time.perf_counter()
        _plot_run_history(history, run_dir, metadata)
        plotting_seconds += time.perf_counter() - plotting_started
    # Reload and rerun the serialized winner; this also catches saving the wrong
    # individual, differing terrain, callback leakage, or inconsistent resets.
    weights = _load_brain(run_dir / "best_brain.json", model)
    score, trace, valid = _simulate(model, mj.MjData(model), weights, args.duration, trace=True)
    if valid != best.tags["valid"] or not np.isclose(score, best.fitness, atol=1e-9, rtol=0):
        raise RuntimeError(f"Saved winner did not reproduce: {score} vs {best.fitness}")
    np.savetxt(run_dir / "trajectory.csv", np.asarray(trace), delimiter=",",
               header="time,x,y,z", comments="")
    result = {"seed": seed, "algorithm": args.algorithm, "fitness": score,
              "valid": valid,
              "evaluations": history[-1]["evaluations"], "stop_reason": stop_reason,
              "new_evaluations": evaluations - inherited_evaluations,
              "generation": ea.current_generation,
              "training_minutes": (time.perf_counter() - started - plotting_seconds) / 60.0,
              "reached_target": _reached_target(score, valid, best.tags, args.target_radius),
              "task_status": dict(best.tags),
              "target_radius": args.target_radius,
              "brain": str((run_dir / "best_brain.json").resolve()),
              "replay_verified": True}
    plotting_started = time.perf_counter()
    _plot_trajectory(trace, run_dir, metadata)
    result["plotting_minutes"] = (plotting_seconds + time.perf_counter() - plotting_started) / 60.0
    _atomic_json(run_dir / "result.json", result)
    return result, history


def _load_brain(path, model):
    """Reject incompatible checkpoints rather than silently changing the task."""
    saved = path if isinstance(path, dict) else json.loads(Path(path).read_text(encoding="utf-8"))
    if saved.get("schema") in (1, 2):
        raise ValueError("This brain uses the old controller inputs. Start fresh without --resume or --replay.")
    if saved.get("schema") != 3:
        raise ValueError("Unsupported saved brain version")
    # Legacy schema 1 mislabeled every world as OlympicArena. Its archived model
    # checksum, not that label, identifies the terrain. New files use the model name.
    # Schema 3 adds elapsed time and heading; older input layouts are rejected.
    expected = {"body": _body_name(), "target": TARGET_POSITION,
                "spawn": SPAWN_POS, "hidden_size": HIDDEN_SIZE,
                "input_size": _input_size(model),
                "output_size": model.nu, "timestep": model.opt.timestep}
    if saved["schema"] >= 2:
        expected["world"] = _world_name(model)
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Checkpoint {key}={saved.get(key)!r}; expected {value!r}")
    # Older schema-3 brains used these fixed values without storing them.
    legacy_controller = {"clock_frequencies": [0.75, 1.5],
                         "position_input_scale": np.pi, "velocity_input_scale": 5.0,
                         "target_input_scale": 2.0, "forward_x_sign": -1.0,
                         "action_angle_limit": np.pi / 2}
    if saved.get("controller_settings", legacy_controller) != _controller_settings():
        raise ValueError("Checkpoint controller settings differ; restore the saved settings to replay/resume.")
    return _decode(saved["genotype"], model)


def _archived_model(brain_path, saved):
    """Load the physics saved with a brain after checking its exact contents."""
    archived_model = Path(brain_path).parent.parent / "model.mjb"
    if hashlib.sha256(archived_model.read_bytes()).hexdigest() != saved["model_sha256"]:
        raise ValueError("Saved model checksum does not match the brain.")
    model = mj.MjModel.from_binary_path(str(archived_model))
    _restore_target(saved, model)
    return model


def _camera():
    """Keep the snake and target together in view, with the stadium behind."""
    camera = mj.MjvCamera()
    camera.lookat[:] = CAMERA_LOOKAT
    camera.distance = CAMERA_DISTANCE
    if AUTO_FRAME_TARGET:
        camera.lookat[:] = (np.asarray(SPAWN_POS) + np.asarray(TARGET_POSITION)) / 2
        camera.distance = max(CAMERA_DISTANCE, 1.5 * np.linalg.norm(
            np.asarray(TARGET_POSITION) - np.asarray(SPAWN_POS)))
    camera.azimuth = CAMERA_AZIMUTH
    camera.elevation = CAMERA_ELEVATION
    return camera


def _interactive_replay(model, data, weights, duration):
    """Replay at real time; hold the end pose briefly, then repeat until closed."""
    mj.set_mjcb_control(None)

    def _control_callback(m, d):
        d.ctrl[:] = nn_controller(m, d, weights, duration)

    camera = _camera()
    total = int(round(duration / model.opt.timestep))
    print("Replay: yellow ball/red pole = target. Close the MuJoCo window to stop.", flush=True)
    try:
        with viewer.launch_passive(model, data) as window:
            window.cam.lookat[:] = camera.lookat
            window.cam.distance = camera.distance
            window.cam.azimuth = camera.azimuth
            window.cam.elevation = camera.elevation
            while window.is_running():
                mj.set_mjcb_control(None)
                mj.mj_resetData(model, data)
                mj.mj_forward(model, data)
                mj.set_mjcb_control(_control_callback)
                start_time = time.perf_counter()
                for step in range(0, total, VIEWER_CHUNK_STEPS):
                    if not window.is_running():
                        return
                    continuing = _advance_with_fall_check(model, data, min(VIEWER_CHUNK_STEPS, total - step))
                    window.sync()
                    time.sleep(max(0, start_time + data.time - time.perf_counter()))
                    if not continuing:
                        break
                score, status = _score_state(model, data)
                print(f"Replay fitness: {score:.6f}; 3D distance: {_format_distance(status['distance_3d'])}; "
                      f"fallen: {status['fallen']}; on terrain: {status['on_terrain']}",
                      flush=True)
                hold_until = time.perf_counter() + REPLAY_HOLD_SECONDS
                while window.is_running() and time.perf_counter() < hold_until:
                    window.sync()
                    time.sleep(VIEWER_POLL_SECONDS)
    finally:
        mj.set_mjcb_control(None)


def _export_video(model, weights, duration, output, *, label=None):
    """Save an independently viewable replay with target and a distance overlay."""
    import cv2
    from PIL import Image

    data = mj.MjData(model)
    mj.set_mjcb_control(None)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    def _control_callback(m, d):
        d.ctrl[:] = nn_controller(m, d, weights, duration)

    fps = VIDEO_FPS
    frame_steps = max(1, int(round(1 / fps / model.opt.timestep)))
    total = int(round(duration / model.opt.timestep))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, VIDEO_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {output}")
    try:
        with mj.Renderer(model, height=VIDEO_SIZE[1], width=VIDEO_SIZE[0]) as renderer:
            mj.set_mjcb_control(_control_callback)
            previous_step = 0
            # Include the exact final state even for durations between frames.
            for step in [*range(0, total, frame_steps), total]:
                continuing = True
                if step > previous_step:
                    continuing = _advance_with_fall_check(model, data, step - previous_step)
                previous_step = step
                renderer.update_scene(data, camera=_camera())
                frame = renderer.render()
                if step == 0:
                    Image.fromarray(frame).save(output.with_suffix(".png"))
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                score, status = _score_state(model, data)
                caption = label or f"{_body_name()} | {_world_name(model)}"
                cv2.putText(bgr, f"{caption} | t={data.time:.1f}s",
                            VIDEO_TEXT_POSITIONS[0], cv2.FONT_HERSHEY_SIMPLEX, VIDEO_TEXT_SCALE,
                            VIDEO_TEXT_COLORS[0], VIDEO_TEXT_THICKNESS, cv2.LINE_AA)
                cv2.putText(bgr, f"3D gap {_format_distance(status['distance_3d'], 3)} | fitness {score:.3f} | "
                            f"{'FALLEN' if status['fallen'] else ('on terrain' if status['on_terrain'] else 'unsupported')}",
                            VIDEO_TEXT_POSITIONS[1], cv2.FONT_HERSHEY_SIMPLEX, VIDEO_TEXT_SCALE,
                            VIDEO_TEXT_COLORS[1], VIDEO_TEXT_THICKNESS, cv2.LINE_AA)
                writer.write(bgr)
                if not continuing:
                    break
            # Hold the final pose for readability without extending the evaluation.
            for _ in range(round(fps * REPLAY_HOLD_SECONDS)):
                writer.write(bgr)
    finally:
        writer.release()
        mj.set_mjcb_control(None)


def _plot_run_history(history, run_dir, metadata):
    """Plot best fitness by generation, with diversity in its own figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return
    generations = np.array([row["generation"] for row in history])
    label = f"{metadata['settings']['algorithm']} | seed {metadata['seed']} | {metadata['world']}"
    _plot_histories([(f"{metadata['settings']['algorithm']} | seed {metadata['seed']}", history)],
                    run_dir / "fitness.png", metadata["world"])

    adaptive = "mean_mutation_sigma" in history[0]
    fig, axes = plt.subplots(3 if adaptive else 2, 1, figsize=(9, 7 if adaptive else 6),
                             sharex=True, layout="constrained")
    axes[0].plot(generations, [row["mean_gene_std"] for row in history], "o-")
    axes[0].set(ylabel="Mean per-weight SD", title=f"Retained population diversity\n{label}")
    axes[1].plot(generations, [row["unique_genotypes"] for row in history], "o-")
    axes[1].set(ylabel="Unique controllers", ylim=(0, metadata["settings"]["population"] + 1))
    if adaptive:
        axes[2].plot(generations, [row["mean_mutation_sigma"] for row in history], "o-")
        axes[2].set(ylabel="Mean mutation sigma")
    axes[-1].set_xlabel("Generation (0 = initial population)")
    for axis in axes:
        axis.grid(alpha=PLOT_GRID_ALPHA)
    fig.savefig(run_dir / "diversity.png", dpi=PLOT_DPI)
    plt.close(fig)


def _plot_trajectory(trace, run_dir, metadata):
    """Show the verified winner's path, progress and height without extra simulation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    trace = np.asarray(trace)
    target = np.asarray(metadata["target"])
    distances = np.linalg.norm(trace[:, 1:4] - target, axis=1)
    planar_distances = np.linalg.norm(trace[:, 1:3] - target[:2], axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    axes[0].plot(trace[:, 1], trace[:, 2], label="Core path")
    axes[0].scatter(*trace[0, 1:3], marker="o", label="Start")
    axes[0].scatter(*trace[-1, 1:3], marker="x", label="End")
    axes[0].scatter(*target[:2], marker="*", s=100, color="red", label="Target")
    axes[0].add_patch(Circle(target[:2], metadata["settings"]["target_radius"],
                             fill=False, color="green", linestyle="--"))
    axes[0].set(xlabel="X (m)", ylabel="Y (m)", title="Winner trajectory (top view)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=8)
    axes[1].plot(trace[:, 0], distances, label="3D distance")
    axes[1].plot(trace[:, 0], planar_distances, linestyle=":", label="XY distance (diagnostic only)")
    axes[1].axhline(metadata["settings"]["target_radius"], color="green", linestyle="--", label="Target radius")
    axes[1].set(xlabel="Simulation time (s)", ylabel="Distance (m)",
                title=f"3D gap: {distances[-1]:.4f} m at t={trace[-1, 0]:.2f}s")
    axes[1].legend(fontsize=8)
    axes[2].plot(trace[:, 0], trace[:, 3], label="Core height")
    axes[2].axhline(target[2], color="green", linestyle="--", label="Target surface height")
    if "fitness_settings" in metadata:
        axes[2].axhline(trace[0, 3] - metadata["fitness_settings"]["fall_height_margin"],
                        color="red", linestyle=":", label="Fall cutoff")
    axes[2].set(xlabel="Simulation time (s)", ylabel="Z (m)", title="Height / falling diagnostic")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=PLOT_GRID_ALPHA)
    fig.suptitle(f"Verified best controller | {metadata['settings']['algorithm']} | seed {metadata['seed']}")
    fig.savefig(run_dir / "trajectory.png", dpi=PLOT_DPI)
    plt.close(fig)


def _plot_histories(histories, output, world_name="selected world"):
    """Plot only each run's best fitness against its actual generation numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axis = plt.subplots(figsize=PLOT_SIZE)
    counts = {}
    for name, rows in histories:
        counts[name] = counts.get(name, 0) + 1
        repeats = sum(label == name for label, _ in histories)
        label = f"{name} | run {counts[name]}" if repeats > 1 else name
        # Plot observed generations only; stopped runs are never extrapolated.
        axis.plot([row["generation"] for row in rows], [row["best"] for row in rows], label=label)
    axis.set(xlabel="Generation", ylabel="Best fitness (lower is better)",
             title=f"{_body_name()} brain evolution in {world_name}")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=PLOT_GRID_ALPHA)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def _aggregate_histories(histories):
    """Average independent runs only at budgets observed in every input run.

    A fixed cohort avoids making a method look better when weak seeds stop.
    Generation plots must also compare the same work at each generation.
    """
    if not histories:
        return {}
    indexed = [(name, {row["evaluations"]: row for row in rows}) for name, rows in histories]
    common = sorted(set.intersection(*(set(rows) for _, rows in indexed)))
    result = {}
    for budget in common:
        generations = {rows[budget]["generation"] for _, rows in indexed}
        if len(generations) != 1:
            raise ValueError("Cross-seed generation plots require matching evaluation budgets per generation")
        for name in sorted({name for name, _ in indexed}):
            scores = [rows[budget]["best"] for label, rows in indexed if label == name]
            result.setdefault(name, []).append({
                "generation": next(iter(generations)), "evaluations": budget, "n": len(scores),
                "mean_best": float(np.mean(scores)),
                "sample_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
            })
    return result


def _plot_comparison(histories, output, world_name="selected world"):
    """Write a separate mean/sample-SD figure and its exact plotted data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    aggregate = _aggregate_histories(histories)
    if not aggregate:
        return
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["algorithm", "generation", "evaluations",
                                                    "n", "mean_best", "sample_std"])
        writer.writeheader()
        for name, rows in aggregate.items():
            writer.writerows({"algorithm": name, **row} for row in rows)
    fig, axis = plt.subplots(figsize=PLOT_SIZE)
    for name, rows in aggregate.items():
        generations = np.asarray([row["generation"] for row in rows])
        means = np.asarray([row["mean_best"] for row in rows])
        line, = axis.plot(generations, means, label=f"{name} | n={rows[0]['n']}")
        if rows[0]["sample_std"] is not None:
            spread = np.asarray([row["sample_std"] for row in rows])
            axis.fill_between(generations, means - spread, means + spread,
                              color=line.get_color(), alpha=PLOT_SPREAD_ALPHA)
    axis.set(xlabel="Generation", ylabel="Mean best fitness (lower is better)",
             title=f"{world_name}: independent runs\nBand = sample SD; common observed budgets only")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=PLOT_GRID_ALPHA)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def _parameter_defaults(path):
    """Read a tuner's portable settings without accepting arbitrary CLI fields."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Parameter file must contain a JSON object")
    settings = payload.get("parameters", payload)
    allowed = {"population", "mutation_rate", "mutation_sigma", "mutation_operator", "crossover_rate",
               "tournament", "search_mode", "duration", "target_radius"}
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ValueError("Parameter file contains unsupported settings")
    for key, value in settings.items():
        if key == "mutation_operator":
            if value not in ("gaussian", "uniform_step", "uniform_reset"):
                raise ValueError("Invalid mutation_operator in parameter file")
        elif key == "search_mode":
            if value not in ("standard", "mixed", "focused", "adaptive"):
                raise ValueError("Invalid search_mode in parameter file")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Invalid numeric parameter {key}")
        elif key in ("population", "tournament") and not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
    return settings


def _cli(model):
    """One command for training, multi-seed studies, or saved-brain replay."""
    parser = argparse.ArgumentParser(description=f"Evolve a {_body_name()} brain in the selected world.",
        epilog="For the controlled mutation comparison, run this file with --mutation-study --help.")
    parser.add_argument("--parameter-file", help="Load tuned settings; explicit command-line flags override them.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--replay", nargs="?", const="latest", help="Replay a brain JSON, or latest.")
    action.add_argument("--resume", nargs="?", const="latest",
                        help="Continue the latest or specified saved population; generations are additional.")
    parser.add_argument("--population", type=int, default=POPULATION_SIZE)
    parser.add_argument("--generations", type=int, default=GENERATIONS, help="Maximum offspring generations.")
    parser.add_argument("--max-minutes", type=float,
                        help="Total training time budget; finish the current generation and save before exiting.")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--duration", type=float, default=SIM_DURATION)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--mutation-rate", type=float, default=MUTATION_RATE)
    parser.add_argument("--mutation-sigma", type=float, default=MUTATION_SIGMA)
    parser.add_argument("--mutation-operator", choices=["gaussian", "uniform_step", "uniform_reset"],
                        default=MUTATION_OPERATOR)
    parser.add_argument("--crossover-rate", type=float, default=CROSSOVER_RATE)
    parser.add_argument("--search-mode", choices=["standard", "mixed", "focused", "adaptive"], default=SEARCH_MODE,
                        help="Adaptive inherits mutation strength per brain and skips crossover; other modes use fixed scales.")
    parser.add_argument("--target-radius", type=float, default=TARGET_RADIUS,
                        help="Success requires this 3D distance plus terrain support/contact; does not change fitness.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Parent folder for numbered body_world runs; defaults to the assignment folder.")
    parser.add_argument("--tournament", type=int, default=TOURNAMENT_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="Stop after this many generations without improvement; 0 disables.")
    parser.add_argument("--min-improvement", type=float, default=MIN_IMPROVEMENT)
    parser.add_argument("--algorithm", choices=["ea", "random"], default=ALGORITHM)
    parser.add_argument("--baseline", action="store_true", default=RUN_BASELINE,
                        help="Run random search at each EA's actual budget.")
    parser.add_argument("--no-view", action="store_true", default=NO_VIEW,
                        help="Do not open the interactive replay.")
    parser.add_argument("--view", action="store_false", dest="no_view", help="Open the interactive replay.")
    parser.add_argument("--video", action="store_true", default=EXPORT_VIDEO,
                        help="Also export best_replay.mp4.")
    parser.add_argument("--no-video", action="store_false", dest="video", help="Skip video export.")
    preliminary, _ = parser.parse_known_args()
    if preliminary.parameter_file:
        try:
            parser.set_defaults(**_parameter_defaults(preliminary.parameter_file))
            task = json.loads(Path(preliminary.parameter_file).read_text(encoding="utf-8")).get("task")
            if task is not None and task != {"target": list(TARGET_POSITION), "body": _body_name(),
                                            "world": _world_name(model), "fitness_settings": _fitness_settings(),
                                            "controller_settings": _controller_settings()}:
                raise ValueError("Tuned parameter file belongs to a different task or controller")
        except (ValueError, OSError) as error:
            parser.error(str(error))
    args = parser.parse_args()
    if args.parameter_file and (args.resume or args.replay or RESUME or REPLAY):
        parser.error("Use a parameter file for fresh training; resume/replay use their archived settings.")
    # An explicit replay/resume flag takes precedence over both file defaults.
    if args.replay is None and args.resume is None:
        args.replay, args.resume = REPLAY, RESUME
    if args.replay and args.resume:
        parser.error("Set only one of REPLAY and RESUME, or choose one on the command line.")
    if args.max_minutes is not None:
        if not np.isfinite(args.max_minutes) or args.max_minutes <= 0:
            parser.error("--max-minutes must be finite and positive.")
        if args.baseline:
            parser.error("A time cap cannot guarantee a matched baseline budget; run the baseline comparison without --max-minutes.")
    if args.resume:
        if args.baseline:
            parser.error("A continued population is not a fresh equal-budget baseline comparison.")
        run = _resolve_run(args.resume, args.output_dir)
        state = _continuation_state(run)
        saved = state["metadata"]
        _require_current_fitness(saved)
        explicitly_set = {argument.split("=", 1)[0] for argument in sys.argv[1:]
                          if argument.startswith("--")}
        # Inherit experiment settings, but let explicit operator flags change the
        # search. --generations is an ADDITIONAL budget; workers are a local choice.
        for key in ("population", "duration", "mutation_rate", "mutation_sigma",
                    "crossover_rate", "tournament", "patience", "min_improvement",
                    "algorithm", "search_mode", "target_radius", "mutation_operator"):
            if "--" + key.replace("_", "-") not in explicitly_set and key in saved["settings"]:
                setattr(args, key, saved["settings"][key])
        if "--seeds" not in explicitly_set:
            args.seeds = [saved["seed"]]
        if args.seeds != [saved["seed"]]:
            parser.error("Continuation retains the original seed; use separate fresh runs for independent seeds.")
        if args.algorithm != saved["settings"]["algorithm"]:
            parser.error("Continuation cannot change the algorithm label.")
        if args.population != len(state["population"]) or args.duration != saved["duration"]:
            parser.error("Continuation must keep the saved population size and simulation duration.")
        if saved["mujoco_version"] != mj.__version__ or saved["numpy_version"] != np.__version__:
            parser.error("Continuation requires the original MuJoCo and NumPy versions.")
        archived_model = run.parent / "model.mjb"
        if hashlib.sha256(archived_model.read_bytes()).hexdigest() != saved["model_sha256"]:
            raise ValueError("Archived model checksum mismatch")
        model = mj.MjModel.from_binary_path(str(archived_model))
        _restore_target(saved, model)
        best_record = min(state["population"], key=lambda record: record["fitness"])
        weights = _load_brain({**saved, "genotype": best_record["genotype"]}, model)
        score, _, valid = _simulate(model, mj.MjData(model), weights, args.duration)
        if valid != best_record["tags"]["valid"] or not np.isclose(score, best_record["fitness"], atol=1e-9, rtol=0):
            raise RuntimeError("Continuation winner does not reproduce its saved score")
        args._continuation = state
        args.resume = str(run)
        kind = "population and RNG restored" if state["rng_state"] is not None else "legacy population restored; new seeded RNG"
        print(f"Continue generation {state['generation']}: {kind}; best fitness={score:.6f}", flush=True)
    if (args.population < 2 or args.generations < 0 or args.workers < 1
            or args.tournament < 1 or args.patience < 0 or args.duration <= 0
            or not np.isfinite(args.duration) or args.mutation_sigma < 0
            or not np.isfinite(args.mutation_sigma) or args.min_improvement < 0
            or not np.isfinite(args.min_improvement)
            or not np.isfinite(args.target_radius) or args.target_radius <= 0
            or not 0 <= args.mutation_rate <= 1 or not 0 <= args.crossover_rate <= 1
            or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Invalid population, budget, seeds, duration, workers, or variation settings.")
    if (args.algorithm == "ea" and args.search_mode in ("focused", "adaptive")
            and (args.mutation_sigma <= 0 or args.mutation_rate <= 0)):
        parser.error("Focused/adaptive search requires positive mutation sigma and rate.")
    if args.mutation_operator != "gaussian" and args.search_mode != "standard":
        parser.error("Alternative mutation operators require --search-mode standard for a controlled comparison.")
    if args.search_mode == "adaptive" and (
        not all(np.isfinite(value) for value in (ADAPTIVE_SIGMA_MIN, ADAPTIVE_SIGMA_MAX, ADAPTIVE_SIGMA_TAU))
        or not 0 < ADAPTIVE_SIGMA_MIN <= ADAPTIVE_SIGMA_MAX or ADAPTIVE_SIGMA_TAU < 0
    ):
        parser.error("Adaptive sigma bounds must be positive and ordered; tau must be finite and nonnegative.")
    if not np.isclose(args.duration / model.opt.timestep,
                      round(args.duration / model.opt.timestep), atol=1e-8, rtol=0):
        parser.error("Duration must be a multiple of the physics timestep (0.002 seconds).")
    if args.replay:
        brain = _resolve_brain(args.replay, args.output_dir)
        saved = json.loads(brain.read_text(encoding="utf-8"))
        # Use the archived binary to reproduce the precise ground and model.
        model = _archived_model(brain, saved)
        weights = _load_brain(brain, model)
        replay_data = mj.MjData(model)
        score, _, valid = _simulate(model, replay_data, weights, saved["duration"])
        same_fitness = saved.get("fitness_settings") == _fitness_settings()
        if same_fitness and (valid != saved["valid"] or not np.isclose(score, saved["fitness"], atol=1e-9, rtol=0)):
            raise RuntimeError("Replay no longer reproduces the stored fitness.")
        status = _task_status(model, replay_data)
        if not same_fitness:
            print(f"Old fitness {saved['fitness']:.6f} is obsolete; re-scored with current terrain-aware fitness.", flush=True)
        print(f"{'Verified' if same_fitness else 'Re-scored'} saved brain: fitness={score:.6f}; "
              f"valid={valid}; 3D distance={_format_distance(status['distance_3d'])}; fallen={status['fallen']}; "
              f"on terrain={status['on_terrain']}; elapsed={replay_data.time:.2f}s", flush=True)
        if not valid:
            print("Numerically invalid controller; diagnostic record retained, animation skipped.", flush=True)
            return
        if args.video:
            _export_video(model, weights, saved["duration"], brain.parent / "best_replay.mp4")
        if not args.no_view:
            _interactive_replay(model, mj.MjData(model), weights, saved["duration"])
        return

    output_root = Path(args.output_dir).resolve() if args.output_dir else DATA
    batch = _new_run_directory(output_root, model)
    model_path = batch / "model.mjb"
    mj.mj_saveModel(model, str(model_path), None)
    # An exact source snapshot makes future experiment interpretation possible.
    (batch / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(f"Results: {batch}\nPopulation {args.population}; at most "
          f"{args.population * (args.generations + (0 if args.resume else 1))} NEW evaluations per run.", flush=True)
    histories, results = [], []
    # Measure real elapsed time for all training runs (including any baselines),
    # excluding the convergence plot, video export, and interactive replay.
    training_started = time.perf_counter()
    if args.max_minutes is not None:
        args._deadline = training_started + args.max_minutes * 60.0
    pool = (ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"),
                                initializer=_worker_init, initargs=(str(model_path), list(TARGET_POSITION)))
            if args.workers > 1 else None)
    if pool is None:
        _worker_init(str(model_path), TARGET_POSITION)
    try:
        for seed in args.seeds:
            if results and time.perf_counter() >= getattr(args, "_deadline", float("inf")):
                break
            run_dir = batch / f"{args.algorithm}_seed_{seed}"
            run_dir.mkdir()
            result, history = _train(model, args, seed, run_dir, pool, model_path)
            results.append(result)
            histories.append((args.algorithm, history))
            if args.baseline and args.algorithm == "ea":
                baseline = argparse.Namespace(**vars(args))
                baseline.algorithm = "random"
                baseline.generations = len(history) - 1
                baseline.patience = 0  # Exact equal budget, even if EA plateaued.
                baseline_dir = batch / f"random_seed_{seed}"
                baseline_dir.mkdir()
                result, history = _train(model, baseline, seed, baseline_dir, pool, model_path)
                results.append(result)
                histories.append(("random", history))
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    training_minutes = ((time.perf_counter() - training_started) / 60.0
                        - sum(result["plotting_minutes"] for result in results))
    print(f"Training finished in {training_minutes:.2f} minutes.", flush=True)
    _plot_histories(histories, batch / "convergence.png", _world_name(model))
    _plot_comparison(histories, batch / "comparison.png", _world_name(model))
    summary = {"runs": results, "statistics": {}, "training_minutes": training_minutes}
    for algorithm in sorted({result["algorithm"] for result in results}):
        runs = [result for result in results if result["algorithm"] == algorithm]
        scores = [result["fitness"] for result in runs]
        summary["statistics"][algorithm] = {"n": len(scores), "mean": float(np.mean(scores)),
            "sample_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
            "success_rate": float(np.mean([result["reached_target"] for result in runs])),
            "valid_rate": float(np.mean([result["valid"] for result in runs])),
            "failure_rate": float(np.mean([not result["valid"] or result["task_status"]["fallen"]
                                            or not result["task_status"]["on_terrain"] for result in runs]))}
    _atomic_json(batch / "summary.json", summary)
    # When a baseline was requested, replay the evolved winner, not a lucky
    # baseline sample. Ties are resolved explicitly by the smaller seed.
    chosen = min((r for r in results if r["algorithm"] == args.algorithm),
                 key=lambda result: (result["fitness"], result["seed"]))
    _atomic_json(output_root / "latest_best.json", chosen)
    _atomic_json(batch / "latest_best.json", chosen)
    brain = Path(chosen["brain"])
    print(f"Best saved brain: {brain}\nVerified fitness: {chosen['fitness']:.6f}; "
          f"valid: {chosen['valid']}; 3D distance: {_format_distance(chosen['task_status']['distance_3d'])}; "
          f"fallen: {chosen['task_status']['fallen']}; on terrain: {chosen['task_status']['on_terrain']}", flush=True)
    print(f"Target within {args.target_radius:.3f} m (3D, supported): "
          f"{'REACHED' if chosen['reached_target'] else 'not reached'}", flush=True)
    if not chosen["valid"]:
        print("No numerically valid controller found; failure results saved, animation skipped.", flush=True)
        return
    if args.video:
        for result in results:
            if result["valid"]:
                run_brain = Path(result["brain"])
                _export_video(model, _load_brain(run_brain, model), args.duration,
                              run_brain.parent / "best_replay.mp4",
                              label=f"{result['algorithm']} seed={result['seed']} | {_body_name()}")
    if not args.no_view:
        _interactive_replay(model, mj.MjData(model), _load_brain(brain, model), args.duration)


if __name__ == "__main__":
    main()


# ============================================================================ #
#  YOUR JOB
# ============================================================================ #
#
# Everything above runs one robot with random weights. It will score badly, and
# it will score badly in a slightly different way every time you change SEED.
# Your task is to replace "random" with "evolved".
#
# Build a proper EA on top of `ariel.ec`. You are expected to use that module -
# it gives you the population/individual data model, the operators, and free
# persistence of every generation to a SQLite database, which you will want
# when it is time to plot convergence curves for the report.
#
#     from ariel.ec import EA, EAOperation, Individual, Population
#
# For a complete, runnable example of how those pieces fit together (a one-max
# EA with parent selection, crossover, mutation and survivor selection written
# as separate steps), read:
#
#     examples/new_EC_engine_example.py
#
# and the API documentation at:
#
#     https://ci-group.github.io/ariel/
#
# ---- EXPERIMENTAL RIGOUR ---------------------------------------------------
#
#   One run proves nothing. Repeat every configuration over several
#     independent seeds and report mean and spread.
#   Log best/mean/worst fitness per generation. The database `ariel.ec`
#     writes makes this straightforward.
#   Compare against a baseline. Random search with the same evaluation
#     budget is a simple, but reasonable choice; and it is nearly free to run.
#   Keep body, world, SIM_DURATION and fitness function identical across
#     everything you compare. Change one thing at a time.
#
# ============================================================================ #
