# Architecture and protocol

## Released representations

For a private image `x`, the two client branches release

```text
r_s = f_s(x) + epsilon_s,    epsilon_s ~ N(0, sigma_s^2 I)
r_t = p(GAP(f_t(x))) + epsilon_t,    epsilon_t ~ N(0, sigma_t^2 I)
```

The G-Path release `r_s` is a `16 x 16 x 16` spatial tensor with 4,096 scalar
values. The S-Path release `r_t` is a 256-dimensional semantic token. The
combined FP32 payload is therefore 4,352 values, or 17,408 bytes per image.

The SlotCEM regulariser is used when training the spatial foundation. It groups
features from different images of the same identity, supported by a bounded
memory bank, and computes a soft within-slot geometric variance. Slots are not
transmitted at deployment: the G-Path releases the original spatial feature map
after Gaussian perturbation.

Global average pooling removes the S-Path's explicit two-dimensional grid, but
it does not establish that identity or appearance information has disappeared.
This branch is evaluated as part of the joint attacker input rather than being
treated as intrinsically private.

## Calibrated fusion

Let `l_s` and `l_t` be the spatial and semantic logits. The deployed classifier
uses

```text
l = (1 - alpha) l_s / tau_s + alpha l_t / tau_t,
```

where `alpha`, `tau_s`, and `tau_t` are learned global parameters. The fusion
rule is shared by all inputs and is not a sample-dependent gate.

## Two-stage training

1. **Semantic warm-up (about 80 epochs).** The spatial client and spatial server
   are frozen. The semantic branch and fusion parameters are trained with
   `sigma_s = 0.025` and `sigma_t = 0.05`.
2. **Deployment adaptation (about 40 epochs).** The protected spatial client
   remains frozen. The spatial server, semantic branch, and fusion parameters
   are updated with `sigma_s = 0.22` and `sigma_t = 0.10`.
3. **Evaluation.** The frozen target is evaluated at `sigma_s = 0.31` and
   `sigma_t = 0.10`.

The final operating point was selected during development and then evaluated
with five adaptation seeds and three attacker initialisations. All adaptations
reuse the same frozen spatial foundation; this is stated explicitly rather than
presented as five independent end-to-end training runs.

## Threat model

The untrusted server, and therefore the attacker, observes both `r_s` and
`r_t`. The retained evidence covers direct decoder, GAN, and adaptive joint
attacks under training-knowledge and inference-knowledge protocols. Privacy is
reported as empirical reconstruction resistance using MSE, PSNR, SSIM, LPIPS,
MAE, and identity cosine similarity. No formal differential-privacy claim is
made.
