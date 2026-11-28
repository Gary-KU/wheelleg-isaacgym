"""Evaluate saved ramp policies while training; never promotes a failed stage."""

import argparse, json, subprocess, time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--height", type=float, required=True)
p.add_argument("--iterations", type=int, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--flat-regression", action="store_true")
p.add_argument("--seeds", type=int, nargs="+", default=(17, 42))
p.add_argument("--interval", type=int, default=50)
p.add_argument("--resume", action="store_true")
p.add_argument("--export-best", type=Path)
p.add_argument("--export-validation-inputs", type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
if a.interval < 1:
    p.error("interval must be positive")
if a.export_best and (not a.flat_regression or len(set(a.seeds)) < 2):
    p.error("export-best requires flat regression and multiple seeds")
seen = set()
results = []
if a.resume and (a.output / "summary.json").exists():
    previous = json.loads((a.output / "summary.json").read_text())
    if previous["height_m"] != a.height:
        raise RuntimeError("Resume validation height differs")
    results = previous["results"]
    if any(
        {x.get("seed") for x in r["reports"]} != set(a.seeds)
        or bool(r.get("flat_regression")) != a.flat_regression
        for r in results
    ):
        raise RuntimeError("Resume validation settings differ")
    seen = {r["iteration"] for r in results}
last_exported = None
while True:
    candidates = sorted(
        a.run.glob("model_*.pt"), key=lambda q: int(q.stem.split("_")[-1])
    )
    for cp in candidates:
        n = int(cp.stem.split("_")[-1])
        if (n % a.interval and n != a.iterations) or n in seen:
            continue
        seen.add(n)
        pair = []
        for seed in a.seeds:
            out = a.output / f"iter{n}_seed{seed}.json"
            with out.with_suffix(".log").open("w") as stream:
                code = subprocess.call(
                    [
                        sys.executable,
                        "sim2sim/evaluate_flight.py",
                        "--checkpoint",
                        str(cp),
                        "--ramp-height",
                        str(a.height),
                        "--policy-only",
                        "--seed",
                        str(seed),
                        "--report",
                        str(out),
                    ],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            pair.append(
                json.loads(out.read_text())["baseline"]
                if code == 0
                else dict(passed=False, error="evaluation failed")
            )
        flat = []
        if a.flat_regression:
            for speed in (0, 1):
                out = a.output / f"iter{n}_flat_speed{speed}.json"
                with out.with_suffix(".log").open("w") as stream:
                    code = subprocess.call(
                        [
                            sys.executable,
                            "sim2sim/evaluate.py",
                            "--checkpoint",
                            str(cp),
                            "--seconds",
                            "8",
                            "--num-envs",
                            "2",
                            "--speed",
                            str(speed),
                            "--report",
                            str(out),
                        ],
                        cwd=ROOT,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                report = (
                    json.loads(out.read_text()) if code == 0 else dict(failures=1000)
                )
                passed = (
                    report["failures"] == 0
                    and max(
                        report.get("pitch_peak_rad", 1e9),
                        report.get("roll_peak_rad", 1e9),
                    )
                    <= 0.2
                )
                if speed == 0:
                    passed = passed and report.get("standing_gate_passed", False)
                else:
                    passed = passed and all(
                        abs(v - 1) < 0.1
                        for v in report.get("settled_forward_speed_m_s", [1e9])
                    )
                flat.append(dict(speed=speed, passed=passed, report=report))
        results.append(
            dict(
                iteration=n,
                checkpoint=str(cp),
                passed=all(x["passed"] for x in pair)
                and all(x["passed"] for x in flat),
                reports=pair,
                flat_regression=flat,
            )
        )

        # Rank the worst seed, rather than automatically selecting the last
        # checkpoint. A later high-reward policy can still regress in playback.
        def score(result):
            reports = result["reports"]
            return (
                not result["passed"],
                sum(x.get("failures", 1000) for x in reports),
                max(
                    max(
                        x.get("tilt_peak_rad", float("inf")) / 0.2,
                        x.get("cross_track_peak_m", float("inf")) / 0.05,
                    )
                    for x in reports
                ),
            )

        best = min(results, key=score)
        (a.output / "best.json").write_text(json.dumps(best, indent=2))
        (a.output / "summary.json").write_text(
            json.dumps(
                dict(
                    height_m=a.height,
                    results=results,
                    stage_passed=any(r["passed"] for r in results),
                    best_checkpoint=best["checkpoint"],
                    full_goal_passed=False,
                ),
                indent=2,
            )
        )
        if a.export_best and best["passed"] and best["checkpoint"] != last_exported:
            command = [
                sys.executable,
                "tools/deployment/export_policy.py",
                "--checkpoint",
                best["checkpoint"],
                "--output",
                str(a.export_best),
                "--require-onnx",
            ]
            if a.export_validation_inputs:
                command += ["--validation-inputs", str(a.export_validation_inputs)]
            with (a.output / "export.log").open("w") as stream:
                code = subprocess.call(
                    command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
                )
            if code == 0:
                manifest = json.loads((a.export_best / "manifest.json").read_text())
                manifest.update(
                    simulation_stage_passed=True,
                    simulation_ramp_height_m=a.height,
                    simulation_seeds=a.seeds,
                )
                for name in ("manifest.json", "metadata.json"):
                    (a.export_best / name).write_text(json.dumps(manifest, indent=2))
                (a.export_best / "validation.json").write_text(
                    json.dumps(
                        dict(
                            checkpoint_sha256=manifest["sha256"],
                            height_m=a.height,
                            result=best,
                        ),
                        indent=2,
                    )
                )
                last_exported = best["checkpoint"]
                print(
                    "Exported simulation-validated candidate:",
                    a.export_best,
                    flush=True,
                )
    if (a.run / f"model_{a.iterations}.pt").exists() and a.iterations in seen:
        break
    time.sleep(5)
print("Ramp validation completed", flush=True)
