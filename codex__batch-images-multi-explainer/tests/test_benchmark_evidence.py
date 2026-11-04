import csv
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark" / "results.csv"


def test_committed_csv_supports_reported_metrics():
    with RESULTS.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 504
    assert len({row["dataset_index"] for row in rows}) == 84
    assert len({row["method_key"] for row in rows}) == 6

    expected = {
        "default_lime": (0.6783761978149414, 51),
        "color_black": (0.8941760046873242, 69),
    }
    for method, (expected_median, expected_changes) in expected.items():
        selected = [row for row in rows if row["method_key"] == method]
        assert len(selected) == 84
        actual_median = median(float(row["cir"]) for row in selected)
        assert math.isclose(actual_median, expected_median, abs_tol=1e-12)
        assert sum(int(row["dir"]) for row in selected) == expected_changes
