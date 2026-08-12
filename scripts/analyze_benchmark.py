#!/usr/bin/env python3
"""Recalculate benchmark metrics and regenerate summary outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "benchmark" / "results.csv"
DEFAULT_OUTPUT = ROOT / "results" / "benchmark"
FIGURE_DIR = ROOT / "assets" / "figures"

EXPECTED = {
    "evaluated_images": 84,
    "default_lime": {
        "median_cir": 0.6783761978149414,
        "dir": 0.6071428571428571,
        "decision_changes": 51,
    },
    "color_black": {
        "median_cir": 0.8941760046873242,
        "dir": 0.8214285714285714,
        "decision_changes": 69,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true", help="Fail if verified metrics drift")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def exact_mcnemar_pvalue(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(0, min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def analyze(rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row["method_key"] in {"default_lime", "color_black"}:
            grouped.setdefault(int(row["dataset_index"]), {})[row["method_key"]] = row

    complete = {index: pair for index, pair in grouped.items() if len(pair) == 2}
    paired_rows: list[dict[str, Any]] = []
    for dataset_index in sorted(complete):
        default = complete[dataset_index]["default_lime"]
        color = complete[dataset_index]["color_black"]
        paired_rows.append(
            {
                "dataset_index": dataset_index,
                "ground_truth_label": default["ground_truth_label"],
                "predicted_label": default["predicted_label"],
                "default_lime_cir": float(default["cir"]),
                "color_lime_cir": float(color["cir"]),
                "cir_difference": float(color["cir"]) - float(default["cir"]),
                "default_lime_dir": int(default["dir"]),
                "color_lime_dir": int(color["dir"]),
            }
        )

    def method_summary(key: str) -> dict[str, Any]:
        selected = [pair[key] for pair in complete.values()]
        cir = [float(row["cir"]) for row in selected]
        decisions = [int(row["dir"]) for row in selected]
        return {
            "n": len(selected),
            "mean_cir": fmean(cir),
            "median_cir": median(cir),
            "dir": fmean(decisions),
            "decision_changes": sum(decisions),
        }

    color_higher = sum(row["cir_difference"] > 0 for row in paired_rows)
    color_lower = sum(row["cir_difference"] < 0 for row in paired_rows)
    color_only = sum(
        row["color_lime_dir"] == 1 and row["default_lime_dir"] == 0
        for row in paired_rows
    )
    default_only = sum(
        row["color_lime_dir"] == 0 and row["default_lime_dir"] == 1
        for row in paired_rows
    )
    both = sum(
        row["color_lime_dir"] == 1 and row["default_lime_dir"] == 1
        for row in paired_rows
    )
    neither = len(paired_rows) - color_only - default_only - both

    metrics = {
        "source_rows": len(rows),
        "evaluated_images": len(paired_rows),
        "default_lime": method_summary("default_lime"),
        "color_black": method_summary("color_black"),
        "paired_comparison": {
            "color_cir_higher": color_higher,
            "color_cir_lower": color_lower,
            "color_only_decision_change": color_only,
            "default_only_decision_change": default_only,
            "both_decision_change": both,
            "neither_decision_change": neither,
            "exact_mcnemar_two_sided_p": exact_mcnemar_pvalue(color_only, default_only),
        },
    }
    return metrics, paired_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify(metrics: dict[str, Any]) -> None:
    assert metrics["evaluated_images"] == EXPECTED["evaluated_images"]
    for method in ("default_lime", "color_black"):
        for key in ("median_cir", "dir"):
            actual = float(metrics[method][key])
            expected = float(EXPECTED[method][key])
            assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), (
                method,
                key,
                actual,
                expected,
            )
        assert metrics[method]["decision_changes"] == EXPECTED[method]["decision_changes"]


def make_plots(metrics: dict[str, Any], paired_rows: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int, bold: bool = False):
        candidates = [
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size)
        return ImageFont.load_default()

    def centered(draw, box, value, chosen_font, fill="#17202A"):
        left, top, right, bottom = box
        bounds = draw.textbbox((0, 0), value, font=chosen_font)
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        draw.text(((left + right - width) / 2, (top + bottom - height) / 2), value, font=chosen_font, fill=fill)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["Default LIME", "Color-LIME Black"]
    colors = ["#6B7A90", "#007C83"]
    medians = [metrics["default_lime"]["median_cir"], metrics["color_black"]["median_cir"]]
    dirs = [100 * metrics["default_lime"]["dir"], 100 * metrics["color_black"]["dir"]]

    canvas = Image.new("RGB", (1890, 756), "white")
    draw = ImageDraw.Draw(canvas)
    centered(draw, (0, 22, 1890, 90), "Color-LIME benchmark — 84 paired ImageNet predictions", font(40, True))
    for panel, (title, values, suffix) in enumerate(
        (("Median confidence impact", medians, ""), ("Decision changes after omission", dirs, "%"))
    ):
        left = 90 + panel * 930
        top, right, bottom = 150, left + 780, 640
        draw.text((left, 105), title, font=font(28, True), fill="#17202A")
        for step in range(6):
            y = bottom - int((bottom - top) * step / 5)
            draw.line((left, y, right, y), fill="#E4E8EB", width=2)
            tick = f"{step / 5:.1f}" if not suffix else f"{step * 20}"
            draw.text((left - 58, y - 13), tick, font=font(20), fill="#52606D")
        draw.line((left, top, left, bottom), fill="#52606D", width=3)
        draw.line((left, bottom, right, bottom), fill="#52606D", width=3)
        for index, (label, value, color) in enumerate(zip(labels, values, colors)):
            x0 = left + 120 + index * 360
            x1 = x0 + 190
            normalized = value if not suffix else value / 100
            y0 = bottom - int((bottom - top) * normalized)
            draw.rounded_rectangle((x0, y0, x1, bottom), radius=8, fill=color)
            display = f"{value:.3f}" if not suffix else f"{value:.1f}%"
            centered(draw, (x0 - 20, y0 - 58, x1 + 20, y0 - 5), display, font(25, True))
            centered(draw, (x0 - 60, bottom + 10, x1 + 60, bottom + 65), label, font(21))
    canvas.save(FIGURE_DIR / "benchmark_summary.png", optimize=True)

    size = 1080
    canvas = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(canvas)
    plot_left, plot_top, plot_right, plot_bottom = 130, 130, 960, 900
    draw.text((130, 40), "Paired confidence impact", font=font(38, True), fill="#17202A")
    for step in range(6):
        x = plot_left + int((plot_right - plot_left) * step / 5)
        y = plot_bottom - int((plot_bottom - plot_top) * step / 5)
        draw.line((x, plot_top, x, plot_bottom), fill="#E4E8EB", width=2)
        draw.line((plot_left, y, plot_right, y), fill="#E4E8EB", width=2)
        draw.text((x - 12, plot_bottom + 16), f"{step / 5:.1f}", font=font(18), fill="#52606D")
        draw.text((plot_left - 58, y - 11), f"{step / 5:.1f}", font=font(18), fill="#52606D")
    draw.line((plot_left, plot_bottom, plot_right, plot_top), fill="#6B7A90", width=3)
    for row in paired_rows:
        x = plot_left + int((plot_right - plot_left) * row["default_lime_cir"])
        y = plot_bottom - int((plot_bottom - plot_top) * row["color_lime_cir"])
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#007C83", outline="white", width=1)
    centered(draw, (plot_left, 925, plot_right, 1000), "Default LIME CIR", font(24))
    draw.text((plot_left + 22, plot_top + 20), "Color-LIME higher: 57/84", font=font(22, True), fill="#007C83")
    canvas.save(FIGURE_DIR / "paired_cir.png", optimize=True)


def main() -> None:
    args = parse_args()
    metrics, paired_rows = analyze(read_rows(args.input))
    if args.verify:
        verify(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "verified_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "paired_default_vs_colorlime.csv", paired_rows)
    write_csv(
        args.output_dir / "headline_summary.csv",
        [
            {"method": "Default LIME", **metrics["default_lime"]},
            {"method": "Color-LIME Black", **metrics["color_black"]},
        ],
    )
    if not args.no_plots:
        make_plots(metrics, paired_rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
