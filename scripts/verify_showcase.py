#!/usr/bin/env python3
"""Validate the frozen public snapshot without private data or checkpoints."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_METRICS = {
    "accuracy": 0.8195617070357555,
    "decoder_inference_mse": 0.03709225594645621,
    "decoder_training_mse": 0.026532018090247737,
    "gan_inference_mse": 0.04039694556279583,
    "gan_training_mse": 0.030993753927236457,
}

EXPECTED_ROWS = {
    "results/formal_attack_runs.csv": 70,
    "results/formal_utility_repeats.csv": 15,
    "results/formal_utility_target_level.csv": 5,
    "results/formal_component_target_level.csv": 5,
    "results/supplementary_attack_runs.csv": 96,
}

REQUIRED_FILES = {
    "README.md",
    "SNAPSHOT.md",
    "DATA.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/ARCHITECTURE.md",
    "docs/privacy_utility.png",
    "results/formal_summary.json",
}

FORBIDDEN_DIRECTORIES = {
    "data",
    "checkpoints",
    "manuscript",
    "results/raw",
}

FORBIDDEN_SUFFIXES = {".ckpt", ".pth", ".pt", ".onnx"}


def csv_rows(relative_path: str) -> int:
    with (ROOT / relative_path).open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    missing = sorted(path for path in REQUIRED_FILES if not (ROOT / path).is_file())
    assert not missing, f"Missing required files: {missing}"

    present_forbidden = sorted(
        path for path in FORBIDDEN_DIRECTORIES if (ROOT / path).exists()
    )
    assert not present_forbidden, f"Private directories included: {present_forbidden}"

    model_files = sorted(
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    )
    assert not model_files, f"Model files included in public snapshot: {model_files}"

    summary = json.loads((ROOT / "results/formal_summary.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["strictly_dominates_published_mean"] is True
    assert summary["confidence_supported_dominance"] is True
    assert summary["target_seeds"] == [126, 127, 128, 129, 130]
    assert summary["attacker_seeds"] == [10125, 20125, 30125]

    for key, expected in EXPECTED_METRICS.items():
        observed = summary["observed"][key]
        assert math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12), (
            f"Unexpected {key}: {observed}"
        )

    for relative_path, expected in EXPECTED_ROWS.items():
        observed = csv_rows(relative_path)
        assert observed == expected, f"Unexpected row count for {relative_path}: {observed}"

    print("Showcase verification passed.")
    print("  Main attack records: 70")
    print("  Supplementary attack records: 96")
    print("  Adaptation seeds: 5")
    print("  Private data/checkpoints: absent")


if __name__ == "__main__":
    main()
