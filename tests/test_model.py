"""Backbone initialisation, LoRA adapter behaviour, freezing. Offline: no weight downloads."""
import argparse

import pytest
import torch
from torch import nn
from torchvision import models

from src.training import train_baseline as tb


def _args(mode, pretrained="false", weights=None, targets="layer4"):
    return argparse.Namespace(mode=mode, pretrained=pretrained, pretrained_weights=weights,
                              lora_rank=4, lora_alpha=8.0, lora_target_layers=targets)


@pytest.mark.parametrize("conv", [nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.Conv2d(16, 32, 1, stride=2),
                                  nn.Conv2d(16, 16, 3, stride=1, padding=1)])
def test_lora_starts_as_identity_and_matches_shape(conv):
    torch.manual_seed(0)
    wrapped = tb.LoRAConv2d(conv, rank=4, alpha=8.0)
    x = torch.randn(2, 16, 15, 15)
    assert torch.allclose(wrapped(x), conv(x), atol=1e-6)          # zero-initialised up-projection
    assert not any(p.requires_grad for p in wrapped.base_conv.parameters())
    assert wrapped.down.weight.requires_grad and wrapped.up.weight.requires_grad


def test_lora_gradients_reach_adapter_only():
    torch.manual_seed(0)
    wrapped = tb.LoRAConv2d(nn.Conv2d(8, 8, 3, padding=1), rank=2, alpha=2.0)
    wrapped(torch.randn(1, 8, 6, 6)).sum().backward()
    assert wrapped.up.weight.grad is not None and wrapped.up.weight.grad.abs().sum() > 0
    assert wrapped.base_conv.weight.grad is None


@pytest.mark.parametrize("bad", [nn.Conv2d(8, 8, 3, padding=0), nn.Conv2d(8, 8, 3, padding=2, dilation=2),
                                 nn.Conv2d(8, 8, 3, padding=1, groups=2)])
def test_lora_rejects_unsupported_geometry(bad):
    with pytest.raises(ValueError):
        tb.LoRAConv2d(bad, rank=2, alpha=2.0)


def test_lora_rank_must_be_positive():
    with pytest.raises(ValueError):
        tb.LoRAConv2d(nn.Conv2d(4, 4, 1), rank=0, alpha=1.0)


def test_trainable_parameter_sets():
    head, init, _ = tb.build_model(_args("head_only"), num_labels=5)
    assert init == "random"
    trainable = {n for n, p in head.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}

    lora, _, layers = tb.build_model(_args("pooled_lora"), num_labels=5)
    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert layers and all(n.startswith("layer4") for n in layers)
    assert trainable - {"fc.weight", "fc.bias"} == {f"{l}.{w}.weight" for l in layers for w in ("down", "up")}
    # 4-rank adapters on layer4 add far fewer parameters than the backbone has
    assert sum(p.numel() for p in lora.parameters() if p.requires_grad) < 0.05 * sum(p.numel() for p in lora.parameters())
    lora.eval()
    assert lora(torch.zeros(1, 3, 64, 64)).shape == (1, 5)


def test_lora_target_without_match_fails():
    with pytest.raises(ValueError, match="No Conv2d modules matched"):
        tb.build_model(_args("pooled_lora", targets="layer9"), num_labels=3)


def test_batchnorm_statistics_stay_frozen_during_training():
    model, _, _ = tb.build_model(_args("head_only"), num_labels=3)
    bn = model.bn1
    before = bn.running_mean.clone()
    tb.set_train_mode(model, True)
    assert model.fc.training and not bn.training
    with torch.no_grad():
        model(torch.randn(4, 3, 64, 64) * 5 + 3)
    assert torch.equal(before, bn.running_mean)


def test_pretrained_failure_is_an_error_not_random_fallback(monkeypatch):
    real = models.resnet50

    def failing(*args, **kwargs):
        if kwargs.get("weights") is not None or kwargs.get("pretrained"):
            raise OSError("simulated: no internet")
        return real(*args, **kwargs)
    monkeypatch.setattr(models, "resnet50", failing)
    with pytest.raises(RuntimeError, match="pretrained"):
        tb.build_resnet50(pretrained=True)


def test_pretrained_from_local_file_is_strict(tmp_path):
    good = tmp_path / "w.pth"
    torch.save(models.resnet50(weights=None).state_dict(), good)
    model, init = tb.build_resnet50(pretrained=True, weights_path=good)
    assert init == "pretrained:file:w.pth"

    bad = tmp_path / "bad.pth"
    state = models.resnet50(weights=None).state_dict()
    state.pop("fc.bias")
    torch.save(state, bad)
    with pytest.raises(RuntimeError):
        tb.build_resnet50(pretrained=True, weights_path=bad)


def test_weights_file_without_pretrained_flag_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        tb.build_resnet50(pretrained=False, weights_path=tmp_path / "w.pth")
