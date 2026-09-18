"""Render the best and worst saved run winners beside their targets in one PNG.

Run from the repository root:
    uv run --locked python assignments/assignment_1/compare_bodies.py
Use --results-dir to select another experiment folder (or a single run).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mujoco
import networkx as nx
import numpy as np

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import load_graph_from_json
from ariel.simulation.environments import SimpleFlatWorld



def find_extremes(results_dir: Path) -> tuple[tuple[Path, float], tuple[Path, float]]:
    """Compare saved run winners; ties are resolved by sorted path order."""
    candidates = []
    for path in sorted(results_dir.rglob("best.json")):
        fitness = float(json.loads(path.read_text(encoding="utf-8"))["fitness"])
        if not math.isfinite(fitness):
            raise ValueError(f"Non-finite fitness in {path}")
        if not path.with_name("best_body.json").is_file():
            raise FileNotFoundError(f"Missing body for {path}")
        candidates.append((path.parent, fitness))
    if not candidates:
        raise FileNotFoundError(f"No saved best.json results under {results_dir}")
    return (min(candidates, key=lambda candidate: candidate[1]),
            max(candidates, key=lambda candidate: candidate[1]))


def prepare_body(graph: nx.DiGraph) -> tuple:
    """Build the saved geometry without running physics or moving the joints."""
    robot = construct_mjspec_from_graph(graph)
    world = SimpleFlatWorld(load_precompiled=False)
    world.spawn(robot.spec, correct_collision_with_floor=True)
    world.spec.visual.global_.offwidth = 1600
    world.spec.visual.global_.offheight = 1600
    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Exclude the floor when framing; bounding spheres include rotated parts.
    mask = model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
    positions = data.geom_xpos[mask]
    radii = model.geom_rbound[mask, None]
    lower = (positions - radii).min(axis=0)
    upper = (positions + radii).max(axis=0)
    return model, data, (lower + upper) / 2, float(np.linalg.norm(upper - lower))


def render_bodies(prepared: list[tuple]) -> list[np.ndarray]:
    """Frame visible robot geometry tightly while preserving scale across bodies."""
    rendered = []
    distance = max(item[3] for item in prepared) * 1.6
    for model, data, center, _ in prepared:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = center
        camera.distance = distance
        camera.azimuth = 135
        camera.elevation = -55
        with mujoco.Renderer(model, width=1600, height=1600) as renderer:
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render().copy()
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render()
            robot_ids = np.flatnonzero(model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
            mask = ((segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
                    & np.isin(segmentation[:, :, 0], robot_ids))
            y, x = np.nonzero(mask)
            rendered.append((rgb, (int(x.min()), int(y.min()), int(x.max()), int(y.max()))))
    # One crop size for every body keeps physical size comparisons valid.
    size = min(1600, math.ceil(max(max(x1 - x0 + 1, y1 - y0 + 1)
                                 for _, (x0, y0, x1, y1) in rendered) * 1.12))
    images = []
    for rgb, (x0, y0, x1, y1) in rendered:
        left = max(0, min(1600 - size, (x0 + x1 - size) // 2))
        top = max(0, min(1600 - size, (y0 + y1 - size) // 2))
        images.append(rgb[top:top + size, left:left + size])
    return images


def body_properties(graph: nx.DiGraph) -> str:
    """Summarize actual modules; NONE nodes are empty slots, not body parts."""
    active = graph.subgraph(
        node for node, attributes in graph.nodes(data=True)
        if attributes["type"] != "NONE"
    )
    counts = Counter(attributes["type"] for _, attributes in active.nodes(data=True))
    root = next(node for node, attributes in active.nodes(data=True)
                if attributes["type"] == "CORE")
    # Depth counts attachments from the core (depth zero) to the farthest part.
    depth = max(nx.single_source_shortest_path_length(active, root).values())
    return (
        f"Modules: {len(active)}  |  Tree depth: {depth}\n"
        f"Cores: {counts['CORE']}  |  Bricks: {counts['BRICK']}  |  Hinges: {counts['HINGE']}"
    )


def create_comparison(results_dir: Path, output: Path) -> Path:
    (run_dir, fitness), (worst_dir, worst_fitness) = find_extremes(results_dir)
    body = load_graph_from_json(run_dir / "best_body.json")
    worst_body = load_graph_from_json(worst_dir / "best_body.json")

    # Use the actual targets saved with the run when available, so a later
    # change to target_bodies cannot silently change the comparison.
    target_snapshot = run_dir / "targets.json"
    if target_snapshot.is_file():
        targets = [
            nx.node_link_graph(item, edges="edges")
            for item in json.loads(target_snapshot.read_text(encoding="utf-8"))
        ]
        labels = [f"Target {index:02d}" for index in range(len(targets))]
    else:
        paths = sorted((HERE / "target_bodies").glob("*.json"))
        targets = [load_graph_from_json(path) for path in paths]
        labels = [path.stem.replace("_", " ").capitalize() for path in paths]
    if not targets:
        raise ValueError("No target bodies found")

    worst_snapshot = worst_dir / "targets.json"
    if worst_snapshot.is_file():
        worst_targets = [
            nx.node_link_graph(item, edges="edges")
            for item in json.loads(worst_snapshot.read_text(encoding="utf-8"))
        ]
        if len(targets) != len(worst_targets) or any(
            not nx.utils.graphs_equal(a, b)
            for a, b in zip(targets, worst_targets)
        ):
            raise ValueError("Selected runs use different targets; compare one experiment")

    colors = ["#087f8c", "#b34d83"]

    def run_label(directory: Path) -> str:
        settings_path = directory / "settings.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.is_file() else {}
        if "k" in settings and "seed" in settings:
            return f"K = {settings['k']}  |  Seed = {settings['seed']}"
        return directory.relative_to(results_dir).as_posix().replace("/", " | ")

    graphs = [body, worst_body, *targets]
    prepared = [prepare_body(graph) for graph in graphs]
    images = render_bodies(prepared)
    image_dir = output.parent / "body_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for name, pixels in zip(["best_body", "worst_body"] +
                            [f"target_{i:02d}" for i in range(len(targets))], images):
        plt.imsave(image_dir / f"{name}.png", pixels)
    # Four panels per row keep the report figure short; center a partial last row.
    rows = math.ceil(len(images) / 4)
    figure = plt.figure(figsize=(12, 3.35 * rows))
    grid = figure.add_gridspec(rows, 8, left=0.015, right=0.985,
                              bottom=0.08, top=0.90, wspace=0.3, hspace=0.52)
    panels = []
    for row in range(rows):
        count = min(4, len(images) - row * 4)
        offset = 4 - count
        for column in range(count):
            start = offset + column * 2
            panels.append(figure.add_subplot(grid[row, start:start + 2]))
    titles = [
        f"Best saved body\nFitness = {fitness:.4f}",
        f"Worst saved body\nFitness = {worst_fitness:.4f}",
    ] + labels
    try:
        for axis, pixels, title, graph in zip(
            panels, images, titles, graphs, strict=True
        ):
            axis.imshow(pixels)
            axis.set_title(title, fontsize=11, fontweight="bold", pad=7)
            axis.text(
                0.5, -0.04, body_properties(graph).split("\n")[0], transform=axis.transAxes,
                ha="center", va="top", fontsize=9, linespacing=1.5,
            )
            axis.axis("off")
        for axis, directory, color in zip(panels[:2], [run_dir, worst_dir], colors):
            axis.title.set_color(color)
            axis.text(0.5, -0.13, run_label(directory), transform=axis.transAxes,
                      ha="center", va="top", fontsize=9, fontweight="bold", color=color)

        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    finally:
        plt.close(figure)
    print(f"Lowest fitness: {fitness:.6f}")
    print(f"Selected run: {run_dir}")
    print(f"Highest saved fitness: {worst_fitness:.6f}")
    print(f"Worst saved run: {worst_dir}")
    print(f"Saved comparison: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results_100_3")
    parser.add_argument("--output", type=Path, default=HERE / "body_comparison.png")
    args = parser.parse_args()
    create_comparison(args.results_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
