#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

try:
    from common import default_output_dir
except ImportError:
    from analysis.common import default_output_dir


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent


def script_path(name: str) -> Path:
    return ANALYSIS_DIR / name


def run_command(cmd: List[str], cwd: Path, continue_on_error: bool) -> int:
    print("\n" + "=" * 88)
    print("RUN:", " ".join(cmd))
    print("=" * 88)

    proc = subprocess.run(cmd, cwd=str(cwd))

    if proc.returncode != 0:
        print(f"[ERROR] Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
        if not continue_on_error:
            raise SystemExit(proc.returncode)

    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the main analysis scripts with outputs under analysis/outputs/."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--real-results-dir", type=Path, default=Path("real_results"))
    parser.add_argument("--html", type=Path, default=Path("results_dashboard.html"))
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--skip-clustering-html", action="store_true")
    parser.add_argument("--skip-quantitative", action="store_true")
    parser.add_argument(
        "--png-only-length",
        action="store_true",
        help="Speed up analyze_length.py by skipping PDFs.",
    )
    parser.add_argument(
        "--skip-heavy-length-scatter",
        action="store_true",
        help="Speed up analyze_length.py by skipping scatter plots.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    real_results = args.real_results_dir
    real_results_abs = real_results if real_results.is_absolute() else repo_root / real_results
    continue_on_error = args.continue_on_error and not args.stop_on_error

    commands: List[List[str]] = [
        [
            sys.executable,
            str(script_path("analyze_conf_bins_project.py")),
            "--repo-root",
            str(repo_root),
            "--real-results-dir",
            str(real_results),
            "--output-name",
            "conf_bin_project_real",
        ],
        [
            sys.executable,
            str(script_path("analyze_conf_decisiveness_gap.py")),
            "--repo-root",
            str(repo_root),
            "--real-results-dir",
            str(real_results),
            "--output-dir",
            str(default_output_dir("confidence_decisiveness_gap_analysis")),
        ],
        [
            sys.executable,
            str(script_path("analyze_high_confidence_wrong_answers.py")),
            "--repo-root",
            str(repo_root),
            "--real-results-dir",
            str(real_results),
            "--output-dir",
            str(default_output_dir("wrong_confidence_percentile_bin_analysis")),
        ],
        [
            sys.executable,
            str(script_path("analyze_length.py")),
            "--root",
            str(real_results_abs),
            "--outdir",
            str(default_output_dir("faithfulness_length_plots_fast")),
        ],
        [
            sys.executable,
            str(script_path("analyze_step_prompt_trajectory_curves.py")),
            "--repo-root",
            str(repo_root),
            "--output-dir",
            str(default_output_dir("step_prompt_faith_conf_curves")),
        ],
        [
            sys.executable,
            str(script_path("analyze_suspicious_traces.py")),
            "--repo-root",
            str(repo_root),
            "--real-results-dir",
            str(real_results),
            "--output-dir",
            str(default_output_dir("trace_signal_analysis")),
        ],
        [
            sys.executable,
            str(script_path("dataset_level_clustering_analysis.py")),
            "--root",
            str(real_results_abs),
            "--outdir",
            str(default_output_dir("clustering_output_dataset_points")),
        ],
        [
            sys.executable,
            str(script_path("generate_dataset_plots.py")),
            "--real-results-dir",
            str(real_results_abs),
            "--output-dir",
            str(default_output_dir("dataset_plots")),
        ],
        [
            sys.executable,
            str(script_path("generate_prompt_plots.py")),
            "--real-results-dir",
            str(real_results_abs),
            "--output-dir",
            str(default_output_dir("prompt_plots")),
        ],
    ]

    if not args.skip_quantitative:
        commands.extend(
            [
                [
                    sys.executable,
                    str(script_path("analyze_quantitative_project.py")),
                    "--repo-root",
                    str(repo_root),
                    "--output-name",
                    "quantitative_project",
                ],
                [
                    sys.executable,
                    str(script_path("analyze_quantitative2_project.py")),
                    "--repo-root",
                    str(repo_root),
                    "--output-dir",
                    str(default_output_dir("quantitative2_project_plots")),
                ],
            ]
        )

    if args.png_only_length:
        for cmd in commands:
            if cmd[1].endswith("analyze_length.py"):
                cmd.append("--png-only")

    if args.skip_heavy_length_scatter:
        for cmd in commands:
            if cmd[1].endswith("analyze_length.py"):
                cmd.append("--skip-scatter")

    html_abs = args.html if args.html.is_absolute() else repo_root / args.html
    if not args.skip_clustering_html and html_abs.exists():
        commands.append(
            [
                sys.executable,
                str(script_path("clustering_analysis.py")),
                "--html",
                str(html_abs),
                "--outdir",
                str(default_output_dir("clustering_output")),
            ]
        )
    elif not args.skip_clustering_html:
        print(f"[SKIP] clustering_analysis.py because {html_abs} does not exist.")

    failures = 0
    for cmd in commands:
        script = Path(cmd[1])
        if not script.exists():
            print(f"[SKIP] Missing script: {script}")
            failures += 1
            if not continue_on_error:
                raise SystemExit(1)
            continue
        failures += int(run_command(cmd, cwd=repo_root, continue_on_error=continue_on_error) != 0)

    print("\n" + "=" * 88)
    if failures:
        print(f"Completed with {failures} failed command(s). Check logs above.")
        raise SystemExit(1)

    print(f"Completed successfully. Outputs are under: {default_output_dir('').parent}")
    print("=" * 88)


if __name__ == "__main__":
    main()
