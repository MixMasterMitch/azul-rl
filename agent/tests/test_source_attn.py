from __future__ import annotations
import pytest
import torch
from agent.env.batched_engine import BatchedEngine
from agent.net.encoder import encode_state
from agent.net.model import AzulNet


@pytest.mark.parametrize('players', [2, 3, 4])
def test_factory_policy_equivariance_and_value_invariance(players: int) -> None:
    torch.manual_seed(321)
    net = AzulNet(hidden=32, arch='source_attn').eval()
    e = BatchedEngine(3, players, 'cpu', seed=41)
    swapped = e.clone()
    swapped.factory_tiles[:, [0, 1]] = e.factory_tiles[:, [1, 0]]
    with torch.no_grad():
        logits, values = net(*encode_state(e), e.legal_action_mask(), players)
        swapped_logits, swapped_values = net(*encode_state(swapped), swapped.legal_action_mask(), players)
    expected = logits.reshape(3, 10, 5, 6).clone()
    expected[:, [0, 1]] = expected[:, [1, 0]]
    assert torch.allclose(swapped_logits, expected.flatten(1), atol=1e-5)
    assert torch.allclose(values, swapped_values, atol=1e-5)
    assert torch.all(logits[~e.legal_action_mask()] == -1e9)


def test_padding_is_invisible_to_source_attn() -> None:
    net = AzulNet(hidden=32, arch='source_attn').eval()
    e = BatchedEngine(2, 2, 'cpu', seed=5)
    g, s = encode_state(e)
    changed = s.clone()
    changed[:, 6:] = 100
    with torch.no_grad():
        p, v = net(g, s, e.legal_action_mask(), 2)
        p2, v2 = net(g, changed, e.legal_action_mask(), 2)
    assert torch.allclose(p, p2, atol=1e-5)
    assert torch.allclose(v, v2, atol=1e-5)


def test_new_policy_tolerates_large_pretrained_source_residuals() -> None:
    torch.manual_seed(19)
    net = AzulNet(hidden=32, arch='source_attn').eval()
    handle = net.source_encoder.register_forward_hook(lambda module, args, output: output * 10000)
    e = BatchedEngine(2, 2, 'cpu', seed=15)
    with torch.no_grad():
        logits, values = net(*encode_state(e), e.legal_action_mask(), 2)
    handle.remove()
    legal_logits = logits[e.legal_action_mask()]
    assert legal_logits.isfinite().all() and values.isfinite().all()
    assert legal_logits.abs().max() < 2
    probs = logits.softmax(1)
    entropy = -(probs * probs.clamp_min(1e-30).log()).sum(1)
    assert entropy.min() > 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA regression')
def test_large_source_attention_has_finite_backward() -> None:
    # Equal, large source tokens stress cancellation when attention reconstructs
    # softmax probabilities from a rounded log-sum-exp in fused CUDA backward.
    # This synthetic case overflowed the old pooling path's gradient norm.
    from agent.net import encoder as enc
    torch.manual_seed(111)
    net = AzulNet(hidden=32, arch='attn', dropout=0).cuda()
    handle = net.source_encoder.register_forward_hook(lambda module, args, output: output * 1e5)
    with torch.no_grad():
        net.attn.in_proj_weight.mul_(100)
        net.attn.out_proj.weight.mul_(100)
    global_feat = torch.randn(2, enc.D_GLOBAL, device='cuda')
    sources = torch.ones(2, enc.NUM_SOURCES, enc.D_SOURCE, device='cuda')
    trunk, _, _ = net._trunk(global_feat, sources, 2)
    (trunk[:, 32:] * torch.arange(32, device='cuda')).sum().backward()
    handle.remove()
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
    assert torch.isfinite(norm)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
