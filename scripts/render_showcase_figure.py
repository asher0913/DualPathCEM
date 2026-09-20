#!/usr/bin/env python3
"""Render the public privacy-utility summary from retained aggregate results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/formal_summary.json"
OUTPUT = ROOT / "docs/privacy_utility.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text())
    published = data["published"]
    observed = data["observed"]
    target_accuracy = [row["accuracy"] for row in data["utility_records"]]
    target_decoder = data["target_attack_means"]["decoder_inference_mse"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.35))
    fig.patch.set_facecolor("#f7f9fb")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.22, wspace=0.28)
    for axis in axes:
        axis.set_facecolor("white")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dce2e8", linewidth=0.8, alpha=0.8)
        axis.tick_params(colors="#44515c")

    left = axes[0]
    left.scatter(
        [published["accuracy"]],
        [published["decoder_inference_mse"]],
        marker="D",
        s=88,
        color="#68737d",
        label="Published CEM",
        zorder=4,
    )
    left.scatter(
        target_accuracy,
        target_decoder,
        s=48,
        color="#7bc8a4",
        edgecolor="white",
        linewidth=0.8,
        label="Adapted targets",
        zorder=3,
    )
    left.scatter(
        [observed["accuracy"]],
        [observed["decoder_inference_mse"]],
        marker="*",
        s=240,
        color="#0d7c66",
        edgecolor="white",
        linewidth=0.9,
        label="DualPath mean",
        zorder=5,
    )
    left.set_title("Privacy-utility operating point", loc="left", weight="bold")
    left.set_xlabel("Top-1 accuracy")
    left.set_ylabel("Decoder inference MSE (higher is safer)")
    left.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    left.set_xlim(0.797, 0.826)
    left.set_ylim(0.018, 0.040)
    left.legend(frameon=False, loc="lower right")

    labels = ["Accuracy", "Decoder\ntrain", "Decoder\ninference", "GAN\ntrain", "GAN\ninference"]
    keys = [
        "accuracy",
        "decoder_training_mse",
        "decoder_inference_mse",
        "gan_training_mse",
        "gan_inference_mse",
    ]
    gains = [100.0 * (observed[key] / published[key] - 1.0) for key in keys]
    colors = ["#2b6cb0"] + ["#0d7c66"] * 4

    right = axes[1]
    bars = right.bar(labels, gains, color=colors, width=0.68)
    right.axhline(0, color="#7a858f", linewidth=0.9)
    right.set_title("Gain over published CEM reference", loc="left", weight="bold")
    right.set_ylabel("Relative improvement (%)")
    right.set_ylim(0, max(gains) + 13)
    for bar, value in zip(bars, gains):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"+{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
            color="#27323a",
        )

    figure_note = (
        "FaceScrub, sigma_s=0.31, sigma_t=0.10. "
        "Five adaptations share one frozen spatial foundation."
    )
    fig.text(0.075, 0.055, figure_note, fontsize=8.5, color="#59636e")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
