# Experiment evidence

This directory contains the sanitised, machine-readable tables retained in the
frozen public showcase. Dataset images, model checkpoints, reconstructed faces,
training logs, failed development trials, and server state are excluded.

## Main evidence

- `formal_summary.json`: audited five-target aggregate and published CEM values.
- `formal_attack_runs.csv`: 70 decoder, GAN, and adaptive attack records.
- `formal_utility_repeats.csv`: 15 noisy-channel utility evaluations.
- `formal_utility_target_level.csv`: one aggregate row per adapted target.
- `formal_component_target_level.csv`: spatial, semantic, and fused utility
  ablations.
- `supplementary_summary.json`: matched-capacity and ResNet-18 controls.
- `supplementary_attack_runs.csv`: 96 supplementary attack records.
- `efficiency_target_level.csv`: parameters, payload, checkpoint size, and GPU
  latency measurements.
- `development_operating_point_screen.json`: the development-only noise screen.

Paths inside some aggregate files are repository-relative provenance labels for
raw records in the private evidence archive. Those raw records are intentionally
not included here; the numerical fields in the retained tables are unchanged.

## Statistical unit

Attacker initialisations are first averaged within each adapted target, and
confidence intervals are then calculated across target seeds. The main method
uses five adaptation seeds. These adaptations share one frozen SlotCEM spatial
foundation, so the interval measures adaptation variability rather than full
end-to-end pretraining variability. The matched-capacity control also uses five
adaptations, while the ResNet-18 control uses three.

## Metric direction

Higher top-1 accuracy is better. Under the evaluated inversion attacks, higher
MSE and lower SSIM or identity similarity indicate a less faithful
reconstruction. These empirical metrics are not a differential-privacy
guarantee. The primary gate compares accuracy and four MSE means with the
published Noise_ARL+CEM row in CEM Table 3.

Run `python scripts/verify_showcase.py` from the repository root to validate the
retained row counts, seeds, headline metrics, and exclusion policy.
