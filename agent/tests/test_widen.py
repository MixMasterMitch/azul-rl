import pytest
import torch

from agent.env.engine import GameEngine
from agent.net.encoder import encode_state
from agent.net.model import AzulNet
from agent.net.widen import transfer_error, widen_source_attention
from agent.train.checkpointing import load_net_from_checkpoint, save_checkpoint


@pytest.mark.parametrize("players", [2, 3, 4])
@pytest.mark.parametrize("factor", [2, 3])
def test_widening_preserves_policy_values_and_attention_heads(
    players: int, factor: int
) -> None:
    torch.manual_seed(17)
    teacher = AzulNet(hidden=32, arch="source_attn").eval()
    # Nontrivial output weights and biases exercise every replicated path.
    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    student = widen_source_attention(teacher, 32 * factor)
    engine = GameEngine(12, players, seed=89)
    for _ in range(7):
        engine.step(engine.legal_action_mask().float().multinomial(1).flatten())
    g, s = encode_state(engine)
    error = transfer_error(teacher, student, g, s, engine.legal_action_mask(), players)
    assert error["argmax_agreement"] == 1
    assert error["max_legal_logit_error"] < 1e-5
    assert error["value_max_error"] < 1e-5
    assert abs(error["policy_kl_mean"]) < 1e-6


def test_widened_network_roundtrips_as_standard_checkpoint_and_learns(tmp_path) -> None:
    torch.manual_seed(7)
    teacher = AzulNet(hidden=32, arch="source_attn").eval()
    student = widen_source_attention(teacher, 64, symmetry_noise=0.01)
    path = tmp_path / "wide.pt"
    save_checkpoint(path, student, config={"num_players": 2})
    loaded, payload = load_net_from_checkpoint(path)
    assert loaded.hidden == 64 and payload["trained_player_counts"] == [2]
    engine = GameEngine(8, 2, seed=1)
    g, s = encode_state(engine)
    legal = engine.legal_action_mask()
    assert transfer_error(teacher, student, g, s, legal)["max_legal_logit_error"] < 1e-5
    assert transfer_error(student, loaded, g, s, legal)["value_max_error"] == 0
    loaded.train()
    opt = torch.optim.AdamW(loaded.parameters(), lr=0.0003)
    old = loaded.g_in[0].weight.detach().clone()
    for _ in range(3):
        opt.zero_grad()
        p, v = loaded(g, s, legal, 2)
        loss = -p.log_softmax(-1)[legal].mean() + v.square().mean()
        loss.backward()
        assert not torch.equal(
            loaded.g_in[0].weight.grad[0], loaded.g_in[0].weight.grad[8]
        )
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(loaded.parameters(), 1.0))
        opt.step()
    assert not torch.equal(old, loaded.g_in[0].weight)
    # The extra channels must become independent during training.
    assert torch.equal(old[0], old[8])
    assert not torch.equal(loaded.g_in[0].weight[0], loaded.g_in[0].weight[8])
    # These heads have no dropout; outgoing zero-sum noise releases their clones.
    assert not torch.equal(
        loaded.policy_heads["0"][0].weight[0], loaded.policy_heads["0"][0].weight[8]
    )


def test_fractional_or_wrong_architecture_expansion_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer"):
        widen_source_attention(AzulNet(hidden=32, arch="source_attn"), 48)
    with pytest.raises(ValueError, match="source_attn"):
        widen_source_attention(AzulNet(hidden=32, arch="flat"), 64)
