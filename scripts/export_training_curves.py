from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_history(trainer_state_path: Path) -> list[dict]:
    data = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    history = data.get("log_history", [])
    if not isinstance(history, list):
        raise ValueError(f"Unexpected log_history in {trainer_state_path}")
    return history


def _extract_points(history: list[dict], metric_name: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in history:
        if metric_name not in row:
            continue
        step = row.get("step")
        if step is None:
            continue
        try:
            points.append((float(step), float(row[metric_name])))
        except (TypeError, ValueError):
            continue
    return points


def _write_csv(path: Path, points: list[tuple[float, float]], metric_name: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", metric_name])
        writer.writerows(points)


def _plot_metric(
    path: Path,
    points: list[tuple[float, float]],
    title: str,
    metric_name: str,
) -> None:
    if not points:
        return

    xs = [step for step, _ in points]
    ys = [value for _, value in points]

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def export_curves(trainer_state_path: Path, out_dir: Path) -> list[Path]:
    history = _load_history(trainer_state_path)
    run_name = trainer_state_path.parent.parent.name
    checkpoint_name = trainer_state_path.parent.name
    prefix = f"{run_name}_{checkpoint_name}"

    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    for metric_name in ("loss", "eval_loss"):
        points = _extract_points(history, metric_name)
        if not points:
            continue

        csv_path = out_dir / f"{prefix}_{metric_name}.csv"
        png_path = out_dir / f"{prefix}_{metric_name}.png"
        title = f"{run_name} / {checkpoint_name} - {metric_name}"

        _write_csv(csv_path, points, metric_name)
        _plot_metric(png_path, points, title, metric_name)
        produced.extend([csv_path, png_path])

    return produced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer_state",
        nargs="+",
        required=True,
        help="Un ou plusieurs fichiers trainer_state.json",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Dossier de sortie pour les CSV/PNG générés",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    produced: list[Path] = []
    for raw_path in args.trainer_state:
        produced.extend(export_curves(Path(raw_path), out_dir))

    for path in produced:
        print(path.as_posix())


if __name__ == "__main__":
    main()