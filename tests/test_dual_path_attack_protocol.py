from PIL import Image

from scripts.attack_dual_path_facescrub import build_attack_loaders
from scripts.combine_dual_path_attack_metrics import combine


def attack_result(knowledge: str, attack_type: str = "decoder") -> dict:
    return {
        "attack_type": attack_type,
        "attack_knowledge": knowledge,
        "dual_path_checkpoint": "target.pt",
        "effective_legacy_noise_std": 0.22,
        "effective_semantic_noise_std": 0.1,
        "evaluation": {"mse": 0.03},
    }


def test_combines_independent_official_attack_splits() -> None:
    result = combine(attack_result("training"), attack_result("inference"))
    assert result["training"]["mse"] == 0.03
    assert result["inference"]["mse"] == 0.03


def test_rejects_swapped_attack_splits() -> None:
    try:
        combine(attack_result("inference"), attack_result("training"))
    except ValueError as error:
        assert "training knowledge" in str(error)
    else:
        raise AssertionError("swapped attack protocols were accepted")


def test_official_split_loader_sizes(tmp_path) -> None:
    for split, count in (("train", 4), ("val", 10)):
        class_dir = tmp_path / split / "identity"
        class_dir.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (8, 8), color=index).save(class_dir / f"{index}.png")

    training_source, training_evaluation = build_attack_loaders(
        tmp_path, batch_size=2, workers=0, attack_knowledge="training"
    )
    inference_source, inference_evaluation = build_attack_loaders(
        tmp_path, batch_size=2, workers=0, attack_knowledge="inference"
    )
    assert len(training_source.dataset) == 4
    assert len(training_evaluation.dataset) == 10
    assert len(inference_source.dataset) == 9
    assert len(inference_evaluation.dataset) == 1
