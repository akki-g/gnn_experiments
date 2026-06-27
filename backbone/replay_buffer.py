import torch

_FIELDS = ("obs", "next_obs", "state", "next_state", "actions", "reward",
           "terminated", "comm_mask", "next_comm_mask")


class ReplayBuffer:
    """Uniform transition replay for feedforward QMIX. Pre-allocated ring buffer.

    Stores the per-step comm_mask AND the terminal (pre-reset) next_obs/next_state/
    next_comm_mask so the TD target bootstraps correctly at fixed-length episode
    boundaries (the QMIX analogue of MAPPO's bad_mask handling)."""

    def __init__(self, capacity, num_agents, obs_dim, state_dim, device="cpu", seed=None):
        self.cap = capacity; self.N = num_agents; self.device = torch.device(device)
        self.obs        = torch.zeros(capacity, num_agents, obs_dim, device=self.device)
        self.next_obs   = torch.zeros(capacity, num_agents, obs_dim, device=self.device)
        self.state      = torch.zeros(capacity, state_dim, device=self.device)
        self.next_state = torch.zeros(capacity, state_dim, device=self.device)
        self.actions    = torch.zeros(capacity, num_agents, dtype=torch.long, device=self.device)
        self.reward     = torch.zeros(capacity, 1, device=self.device)   # team scalar
        self.terminated = torch.zeros(capacity, 1, device=self.device)   # true termination only
        self.comm_mask      = torch.zeros(capacity, num_agents, num_agents, dtype=torch.bool, device=self.device)
        self.next_comm_mask = torch.zeros(capacity, num_agents, num_agents, dtype=torch.bool, device=self.device)
        self.ptr = 0; self.full = False
        # Dedicated sampler RNG so replay sampling is reproducible and independent of
        # the global torch RNG (which also drives epsilon-greedy exploration).
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(int(seed))

    def insert_batch(self, **kw):
        """Insert B transitions at once (vectorized env step). Each value has leading dim B."""
        missing = set(_FIELDS) - set(kw)
        assert not missing, f"insert_batch missing required keys: {sorted(missing)}"
        B = kw["reward"].shape[0]
        assert B <= self.cap, f"batch B={B} exceeds buffer capacity {self.cap}"
        idx = (torch.arange(B, device=self.device) + self.ptr) % self.cap
        for name in _FIELDS:
            assert kw[name].shape[0] == B, (
                f"insert_batch '{name}' leading dim {kw[name].shape[0]} != B {B}")
            getattr(self, name)[idx] = kw[name].to(self.device)
        self.ptr = (self.ptr + B) % self.cap
        self.full = self.full or self.ptr < B  # wrapped (correct for all B <= cap)
        return

    def __len__(self):
        return self.cap if self.full else self.ptr

    def sample(self, batch_size):
        n = len(self)
        idx = torch.randint(0, n, (batch_size,), device=self.device, generator=self.gen)
        return {name: getattr(self, name)[idx] for name in _FIELDS}
