"""Controlled mutation study for Assignment 2; see README_mutation_study.md.

Only mutation operator/rate/strength differ between EA conditions. Every
condition and the random-search baseline use the same seeds and search budget.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import A2_template_2026 as a


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _conditions(operators, rates, sigmas):
    conditions = []
    for operator in operators:
        for rate in rates:
            # A random reset has no step-size parameter: do not duplicate it.
            for sigma in ([None] if operator == "uniform_reset" else sigmas):
                name = f"{operator}_p{rate}" + (f"_s{sigma}" if sigma is not None else "")
                conditions.append(dict(id=name, algorithm="ea", mutation_operator=operator,
                                       mutation_rate=rate, mutation_sigma=sigma))
    conditions.append(dict(id="random_search", algorithm="random", mutation_operator="gaussian",
                           mutation_rate=0.0, mutation_sigma=0.0))
    return conditions


def _validate(args):
    if args.population < 2 or args.generations < 1 or args.workers < 1 or args.tournament < 1:
        raise ValueError("Population >=2, generations >=1, workers >=1 and tournament >=1 required")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError("Seeds must be distinct and nonnegative")
    if len(args.seeds) < 5 and not args.smoke:
        raise ValueError("The assignment requires at least five independent seeds; use --smoke for pipeline checks")
    for values in (args.operators, args.rates, args.sigmas):
        if len(values) != len(set(values)):
            raise ValueError("Duplicate mutation values would duplicate conditions")
    if any(not 0 <= rate <= 1 for rate in args.rates):
        raise ValueError("Mutation rates must be in [0,1]")
    if any(not a.np.isfinite(sigma) or sigma <= 0 for sigma in args.sigmas):
        raise ValueError("Mutation strengths must be positive and finite")
    if not 0 <= args.crossover_rate <= 1:
        raise ValueError("Crossover rate must be in [0,1]")
    if not a.np.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("Duration must be positive and finite")
    if args.max_minutes is not None and (not a.np.isfinite(args.max_minutes) or args.max_minutes <= 0):
        raise ValueError("Time cap must be positive and finite")


def _save(root, state):
    a._atomic_json(root / "study.json", state)


def _trial(root, state, condition, seed, model, pool, workers, deadline):
    plan = state["plan"]
    budget = plan["population"] * (plan["generations"] + 1)
    key = f"{condition['id']}_seed_{seed}"
    trial = state["trials"].setdefault(key, dict(condition=condition["id"], seed=seed, attempts=[]))
    if trial.get("complete"):
        return True
    continuation, prior = None, None
    for relative in reversed(trial["attempts"]):
        prior = root / relative
        if (prior / "checkpoint.json").exists():
            continuation = a._continuation_state(prior)
            if continuation["evaluations"] == budget and (prior / "result.json").exists():
                result = _read(prior / "result.json")
                result["brain"] = str((prior / "best_brain.json").relative_to(root))
                trial.update(complete=True, result=result)
                _save(root, state)
                return True
            break
    if time.perf_counter() >= deadline:
        return False
    inherited = continuation["evaluations"] if continuation else 0
    generations = (budget - inherited) // plan["population"] - (0 if continuation else 1)
    if generations < 0:
        raise ValueError("Checkpoint exceeds declared search budget")
    batch = root / "runs" / key / f"attempt_{len(trial['attempts']):03d}"
    batch.mkdir(parents=True)
    shutil.copy2(root / "model.mjb", batch / "model.mjb")
    run = batch / "run"
    run.mkdir()
    args = argparse.Namespace(population=plan["population"], generations=generations,
                              duration=plan["duration"], workers=workers,
                              tournament=plan["tournament"], crossover_rate=plan["crossover_rate"],
                              search_mode="standard", algorithm=condition["algorithm"],
                              mutation_operator=condition["mutation_operator"],
                              mutation_rate=condition["mutation_rate"],
                              mutation_sigma=condition["mutation_sigma"] or 0.0,
                              patience=0, min_improvement=a.MIN_IMPROVEMENT,
                              target_radius=plan["target_radius"], resume=None, _deadline=deadline)
    if continuation:
        best = min(continuation["population"], key=lambda row: row["fitness"])
        weights = a._load_brain({**continuation["metadata"], "genotype": best["genotype"]}, model)
        score, _, valid = a._simulate(model, a.mj.MjData(model), weights, args.duration)
        if valid != best["tags"]["valid"] or not a.np.isclose(score, best["fitness"], atol=1e-9, rtol=0):
            raise RuntimeError("Checkpoint winner no longer reproduces")
        args._continuation, args.resume = continuation, str(prior)
    trial["attempts"].append(str(run.relative_to(root)))
    _save(root, state)
    print(f"\n{key}: {inherited}/{budget} evaluations", flush=True)
    result, _ = a._train(model, args, seed, run, pool, batch / "model.mjb")
    # Use a study-relative location so moving the entire archive is supported.
    result["brain"] = str((run / "best_brain.json").relative_to(root))
    trial.update(result=result, complete=result["evaluations"] == budget)
    _save(root, state)
    return trial["complete"]


def _history(root, trial):
    """Merge resume attempts, discarding history newer than a committed checkpoint."""
    rows = {}
    for relative in trial["attempts"]:
        run = root / relative
        if not (run / "checkpoint.json").exists():
            continue
        committed = _read(run / "checkpoint.json")["evaluations"]
        with (run / "history.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                point = {key: float(value) for key, value in row.items()}
                if point["evaluations"] <= committed:
                    rows[int(point["generation"])] = point
    return [rows[key] for key in sorted(rows)]


def _cohort(state):
    """Only complete paired seed blocks enter comparisons, including the baseline."""
    return [seed for seed in state["plan"]["seeds"] if all(
        state["trials"].get(f"{c['id']}_seed_{seed}", {}).get("complete")
        for c in state["plan"]["conditions"])]


def _report(root, state):
    seeds = _cohort(state)
    plan = state["plan"]
    names = [c["id"] for c in plan["conditions"]]
    summary = dict(status=state["status"], smoke=plan["smoke"], paired_seeds=seeds,
                   planned_seeds=plan["seeds"], evaluations_per_run=plan["population"] * (plan["generations"] + 1),
                   completed_runs=sum(t.get("complete", False) for t in state["trials"].values()),
                   planned_runs=len(names) * len(plan["seeds"]), statistics={})
    raw = []
    for trial in state["trials"].values():
        if trial.get("complete"):
            r = trial["result"]
            raw.append(dict(condition=trial["condition"], seed=trial["seed"], fitness=r["fitness"],
                            distance_3d=r["task_status"]["distance_3d"], success=r["reached_target"],
                            valid=r["valid"], fallen=r["task_status"]["fallen"],
                            on_terrain=r["task_status"]["on_terrain"], evaluations=r["evaluations"],
                            included_in_paired_summary=trial["seed"] in seeds, brain=r["brain"]))
    if raw:
        with (root / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(raw[0]))
            writer.writeheader()
            writer.writerows(raw)
    lines = ["# Mutation study", "", f"Status: {state['status']}. Completed runs: {summary['completed_runs']}/{summary['planned_runs']}.",
             f"Paired complete seeds used below: {seeds}. Planned: {plan['seeds']}.",
             "SMOKE TEST ONLY: no scientific performance conclusion." if plan["smoke"] else
             "Results describe this fixed body, arena and budget; they do not establish a generally best mutation.", "",
             "Mean +/- sample standard deviation across independent seeds; lower fitness is better.",
             "Unfinished or unmatched seeds are excluded from comparisons, never padded forward.", "",
             "| Condition | n | Mean fitness | Sample SD | Success | Failure |", "|---|---:|---:|---:|---:|---:|"]
    if seeds:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
        diag, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        final, ends = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
        base = a.np.array([state["trials"][f"random_search_seed_{s}"]["result"]["fitness"] for s in seeds])
        final_scores, successes, failures = [], [], []
        colors = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
        for index, name in enumerate(names):
            trials = [state["trials"][f"{name}_seed_{s}"] for s in seeds]
            histories = [_history(root, trial) for trial in trials]
            generations = [row["generation"] for row in histories[0]]
            expected = list(range(plan["generations"] + 1))
            if any([row["generation"] for row in history] != expected for history in histories):
                raise ValueError("Completed run has a missing generation; comparison would be misleading")
            scores = a.np.array([t["result"]["fitness"] for t in trials])
            success = a.np.mean([t["result"]["reached_target"] for t in trials])
            failure = a.np.mean([not t["result"]["valid"] or t["result"]["task_status"]["fallen"]
                                or not t["result"]["task_status"]["on_terrain"] for t in trials])
            std = float(scores.std(ddof=1)) if len(seeds) > 1 else None
            summary["statistics"][name] = dict(n=len(seeds), mean=float(scores.mean()), sample_std=std,
                success_rate=float(success), failure_rate=float(failure),
                paired_difference_from_random=(scores - base).tolist(),
                mean_difference_from_random=float((scores - base).mean()))
            lines.append(f"| {name} | {len(seeds)} | {scores.mean():.6f} | {std if std is not None else 'n/a'} | {success:.0%} | {failure:.0%} |")
            color = colors(index % colors.N)
            label = name.replace("_", " ")
            for axis, metric in ((ax, "best"), (axes[0], "mean_gene_std"), (axes[1], "unique_genotypes")):
                values = a.np.array([[row[metric] for row in history] for history in histories])
                mean = values.mean(axis=0)
                axis.plot(generations, mean, label=label, color=color)
                if len(seeds) > 1:
                    spread = values.std(axis=0, ddof=1)
                    axis.fill_between(generations, mean - spread, mean + spread, alpha=0.13, color=color)
            final_scores.append(scores)
            successes.append(success)
            failures.append(failure)
            ends[0].scatter(a.np.full(len(seeds), index), scores, color=color, zorder=3)
            ends[0].errorbar(index, scores.mean(), yerr=std or 0, color="black", fmt="_", capsize=4)
        ax.set(xlabel="Offspring generation (0 = initial population)", ylabel="Best fitness (lower is better)",
               title=f"Mutation comparison: mean +/- sample SD, n={len(seeds)} paired seeds")
        top = ax.secondary_xaxis("top", functions=(lambda g: (g + 1) * plan["population"],
                                                   lambda e: e / plan["population"] - 1))
        top.set_xlabel("Search evaluations per run (verification/video excluded)")
        ax.legend(fontsize=8, loc="best")
        axes[0].set(ylabel="Mean per-gene population SD", title="Genotype diversity")
        axes[1].set(ylabel="Unique surviving genotypes", title="Population duplication")
        for axis in axes:
            axis.set_xlabel("Offspring generation")
        axes[0].legend(fontsize=7)
        # Each grey line joins the same seed across conditions.
        for paired in a.np.array(final_scores).T:
            ends[0].plot(range(len(names)), paired, color="grey", alpha=0.3, linewidth=0.7)
        ends[0].set(title="Final best fitness: paired seeds, mean +/- SD", ylabel="Fitness (lower is better)")
        x = a.np.arange(len(names))
        success_bars = ends[1].bar(x - 0.18, successes, width=0.36, label="Target reached")
        failure_bars = ends[1].bar(x + 0.18, failures, width=0.36, label="Invalid / fallen / off terrain")
        ends[1].bar_label(success_bars, labels=[f"{value:.0%}" for value in successes], fontsize=7)
        ends[1].bar_label(failure_bars, labels=[f"{value:.0%}" for value in failures], fontsize=7)
        ends[1].set(title="Best-controller outcomes", ylabel="Fraction of independent runs", ylim=(0, 1.05))
        ends[1].legend(fontsize=8)
        for axis in ends:
            axis.set_xticks(x, names, rotation=35, ha="right", fontsize=8)
        for axis in (ax, *axes, *ends):
            axis.grid(alpha=0.2, axis="y")
        for axis in (ax, *axes):
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        for figure, filename in ((fig, "convergence"), (diag, "diversity"), (final, "final_results")):
            if plan["smoke"]:
                figure.suptitle("PIPELINE SMOKE TEST - not research evidence", color="darkred")
            figure.savefig(root / f"{filename}.png", dpi=170)
            figure.savefig(root / f"{filename}.svg")
            plt.close(figure)
    else:
        lines += ["", "No complete paired seed block yet; comparison graphs will appear after one completes."]
    lines += ["", "Fitness is 3D final core-to-target distance plus terrain/fall penalties (fixed for all conditions).",
              "The assignment's default is planar distance; explain and justify this alternative in Methods.",
              "Inspect convergence before selecting the final budget. Extend all conditions equally if still improving.",
              "Video representatives are each run's best controller; they are not additional independent runs."]
    a._atomic_json(root / "summary.json", summary)
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _videos(root, state, model):
    """Render outside search timing; a failed export is retryable without retraining."""
    errors = []
    for key, trial in state["trials"].items():
        if not trial.get("complete"):
            continue
        if not trial["result"]["valid"]:
            trial["video_status"] = "skipped: numerically invalid controller"
            continue
        output = root / "videos" / f"{key}.mp4"
        if trial.get("video_status") == "saved" and output.exists() and output.stat().st_size > 0:
            continue
        output.parent.mkdir(exist_ok=True)
        temporary = output.with_name(output.stem + ".partial.mp4")
        try:
            brain = root / trial["result"]["brain"]
            weights = a._load_brain(brain, model)
            a._export_video(model, weights, state["plan"]["duration"], temporary,
                            label=f"{trial['condition']} | seed={trial['seed']}")
            import cv2
            capture = cv2.VideoCapture(str(temporary))
            try:
                ok, frame = capture.read()
                if not ok or frame is None or capture.get(cv2.CAP_PROP_FRAME_COUNT) < 2:
                    raise RuntimeError("Exported video could not be decoded")
            finally:
                capture.release()
            temporary.replace(output)
            temporary.with_suffix(".png").replace(output.with_suffix(".png"))
            trial["video_status"] = "saved"
            trial.pop("video_error", None)
            print(f"Video: {output}", flush=True)
        except Exception as error:
            trial.update(video_status="failed", video_error=str(error))
            errors.append(key)
            print(f"Video failed for {key}: {error}", flush=True)
        _save(root, state)
    _save(root, state)
    return errors


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", nargs="+", choices=["gaussian", "uniform_step", "uniform_reset"],
                        default=["gaussian", "uniform_reset"])
    parser.add_argument("--rates", type=float, nargs="+", default=[0.1])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.05, 0.15, 0.45])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--population", type=int, default=a.POPULATION_SIZE)
    parser.add_argument("--generations", type=int, default=a.GENERATIONS)
    parser.add_argument("--duration", type=float, default=a.SIM_DURATION)
    parser.add_argument("--tournament", type=int, default=a.TOURNAMENT_SIZE)
    parser.add_argument("--crossover-rate", type=float, default=a.CROSSOVER_RATE)
    parser.add_argument("--workers", type=int, default=a.WORKERS)
    parser.add_argument("--max-minutes", type=float, help="Pause training at a generation boundary; resume the same plan later")
    parser.add_argument("--output-dir", type=Path, help="New study directory (must not already exist)")
    parser.add_argument("--resume", type=Path, help="Existing study directory; use only workers/time-cap/video flags with it")
    parser.add_argument("--no-video", action="store_true", help="Defer video export; resume later to render without retraining")
    parser.add_argument("--smoke", action="store_true", help="Allow <5 seeds and label all outputs as pipeline checks")
    args = parser.parse_args(argv)
    tokens = list(argv) if argv is not None else __import__("sys").argv[1:]
    try:
        if args.resume:
            allowed = {"--resume", "--workers", "--max-minutes", "--no-video"}
            if any(t.split("=", 1)[0] not in allowed for t in tokens if t.startswith("--")):
                raise ValueError("Resume keeps the saved plan; only workers, max-minutes and no-video may change")
        _validate(args)
    except ValueError as error:
        parser.error(str(error))
    sources = {"A2_template_2026.py": Path(a.__file__), "mutation_study_A2_2026.py": Path(__file__)}
    if args.resume:
        root = args.resume.resolve()
        state = _read(root / "study.json")
        if state.get("schema") != 1:
            raise ValueError("Unsupported study schema")
        if state["versions"] != dict(mujoco=a.mj.__version__, numpy=a.np.__version__):
            raise ValueError("Resume requires the original MuJoCo and NumPy versions")
        for name, path in sources.items():
            if state["source_hashes"][name] != _hash(path):
                raise ValueError(f"Source changed: {name}. Resume using the archived source files or start a fresh study")
        if state["model_sha256"] != _hash(root / "model.mjb"):
            raise ValueError("Archived model checksum mismatch")
        a.mj.set_mjcb_control(None)
        model = a.mj.MjModel.from_binary_path(str(root / "model.mjb"))
        a.TARGET_POSITION[:] = state["target"]
    else:
        root = (args.output_dir or a.DATA / "mutation_studies" / time.strftime("study_%Y%m%d_%H%M%S")).resolve()
        world = a.build_world()
        world.spawn(a.build_robot().spec, position=a.SPAWN_POS, rotation=a.SPAWN_ROTATION,
                    correct_collision_with_floor=a.CORRECT_SPAWN_COLLISION)
        model = world.spec.compile()
        if not a.np.isclose(args.duration / model.opt.timestep, round(args.duration / model.opt.timestep), atol=1e-8, rtol=0):
            parser.error("Duration must be a multiple of the physics timestep")
        root.mkdir(parents=True, exist_ok=False)
        a.mj.mj_saveModel(model, str(root / "model.mjb"), None)
        for name, path in sources.items():
            shutil.copy2(path, root / name)
        plan = {key: getattr(args, key) for key in ("population", "generations", "duration", "tournament", "crossover_rate", "seeds", "smoke")}
        plan.update(target_radius=a.TARGET_RADIUS, conditions=_conditions(args.operators, args.rates, args.sigmas),
                    search_mode="standard", survivor_selection="elitist mu+lambda, duplicates allowed",
                    patience=0, terrain_seed=a.TERRAIN_SEED)
        state = dict(schema=1, plan=plan, status="running", target=list(a.TARGET_POSITION), trials={},
                     model_sha256=_hash(root / "model.mjb"),
                     source_hashes={name: _hash(path) for name, path in sources.items()},
                     versions=dict(mujoco=a.mj.__version__, numpy=a.np.__version__))
    state["status"] = "running"
    _save(root, state)
    plan = state["plan"]
    print(f"Study: {root}\n{len(plan['conditions'])} conditions x {len(plan['seeds'])} seeds; "
          f"{plan['population'] * (plan['generations'] + 1)} search evaluations per run", flush=True)
    deadline = time.perf_counter() + args.max_minutes * 60 if args.max_minutes else float("inf")
    pool = (ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"),
            initializer=a._worker_init, initargs=(str(root / "model.mjb"), state["target"])) if args.workers > 1 else None)
    if pool is None:
        a._worker_init(str(root / "model.mjb"), state["target"])
    try:
        complete = True
        for seed in plan["seeds"]:
            for condition in plan["conditions"]:
                if not _trial(root, state, condition, seed, model, pool, args.workers, deadline):
                    complete = False
                    break
            _report(root, state)
            if not complete:
                break
        state["status"] = "complete" if complete else "paused"
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        print("Interrupted; resume from the last committed generation.", flush=True)
    except Exception:
        state["status"] = "failed"
        raise
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        _save(root, state)
        _report(root, state)
    errors = [] if args.no_video else _videos(root, state, model)
    print(f"Study {state['status']}; report: {root / 'REPORT.md'}", flush=True)
    if errors:
        raise RuntimeError("Some videos failed. Training is saved; --resume retries their exports")


if __name__ == "__main__":
    _main()
