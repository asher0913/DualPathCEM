import torch

from publication_cem.dual_path_attack import DualPathReconstructor, PatchDiscriminator


def test_dual_path_reconstructor_outputs_image() -> None:
    model = DualPathReconstructor(width=32, residual_blocks=2)
    images = model(torch.randn(2, 16, 16, 16), torch.randn(2, 256))
    assert images.shape == (2, 3, 64, 64)
    assert torch.all((0 <= images) & (images <= 1))


def test_dual_path_reconstructor_backpropagates() -> None:
    model = DualPathReconstructor(width=32, residual_blocks=1)
    model(torch.randn(1, 16, 16, 16), torch.randn(1, 256)).mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_patch_discriminator_outputs_batch_scores() -> None:
    discriminator = PatchDiscriminator(width=16)
    scores = discriminator(torch.rand(3, 3, 64, 64))
    assert scores.shape[0] == 3
