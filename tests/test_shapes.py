"""
Shape contract tests for encoder, actor, critic, and identity comm.

Tests per planner §12.
"""

import torch
import pytest

from backbone.encoder import SharedMLPEncoder
from backbone.actor import Actor
from backbone.critic import CentralCritic
from comm import IdentityComm


# -------------------------------------------------------------------
# Fixtures: small dimensions
# -------------------------------------------------------------------
B, N, obs_dim, hidden_dim, action_dim, state_dim = 4, 2, 6, 32, 3, 12


def test_encoder_output_shape():
    """Encoder maps [B, N, obs_dim] → [B, N, hidden_dim]."""
    enc = SharedMLPEncoder(obs_dim=obs_dim, hidden_dim=hidden_dim, num_layers=2)
    obs = torch.randn(B, N, obs_dim)
    h = enc(obs)
    assert h.shape == (B, N, hidden_dim), f"expected {(B, N, hidden_dim)}, got {h.shape}"


def test_actor_output_shape():
    """Actor.act returns action [B, N], log_prob [B, N], entropy [B, N]."""
    actor = Actor(hidden_dim=hidden_dim, action_dim=action_dim)
    h = torch.randn(B, N, hidden_dim)
    action, log_prob, entropy = actor.act(h)
    assert action.shape == (B, N), f"action shape {action.shape}"
    assert log_prob.shape == (B, N), f"log_prob shape {log_prob.shape}"
    assert entropy.shape == (B, N), f"entropy shape {entropy.shape}"


def test_actor_logprob_shape():
    """Actor.evaluate_actions returns log_probs [B, N] and entropy [B, N]."""
    actor = Actor(hidden_dim=hidden_dim, action_dim=action_dim)
    h = torch.randn(B, N, hidden_dim)
    actions = torch.randint(0, action_dim, (B, N))
    log_probs, entropy = actor.evaluate_actions(h, actions)
    assert log_probs.shape == (B, N), f"log_probs shape {log_probs.shape}"
    assert entropy.shape == (B, N), f"entropy shape {entropy.shape}"


def test_critic_value_shape():
    """Critic maps [B, state_dim] → [B, num_agents]."""
    critic = CentralCritic(state_dim=state_dim, hidden_dim=hidden_dim, num_agents=N)
    state = torch.randn(B, state_dim)
    values = critic(state)
    assert values.shape == (B, N), f"values shape {values.shape}"


def test_identity_comm_preserves_shape():
    """IdentityComm returns the exact same tensor, shape preserved."""
    comm = IdentityComm()
    h = torch.randn(B, N, hidden_dim)
    out = comm(h)
    assert out is h, "IdentityComm must return the same tensor object"
    assert out.shape == (B, N, hidden_dim), f"shape changed: {out.shape}"
