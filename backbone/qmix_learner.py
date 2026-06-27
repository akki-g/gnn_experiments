import copy
import torch
import torch.nn as nn

from backbone.q_agent import QAgent
from backbone.mixer import QMixer


class QMIXLearner:
    """
    QMIX double-Q TD update.

    Live (agent, mixer) are trained; target (agent_t, mixer_t) provide the bootstrap.
    Double-Q: next action is SELECTED by the live agent (argmax) but its VALUE is read
    from the TARGET agent (decorrelation -> less overestimation). The mixer reads RAW
    state only (isolation guarantee).
    """

    def __init__(self, agent: QAgent, mixer: QMixer, class_id, *,
                 lr, gamma, grad_clip, double_q=True, device="cpu"):
        self.agent = agent
        self.mixer = mixer
        # target networks (frozen copies, synced periodically by sync_targets())
        self.agent_t = copy.deepcopy(agent).eval()
        self.mixer_t = copy.deepcopy(mixer).eval()

        for p in self.agent_t.parameters():
            p.requires_grad_(False)
        for p in self.mixer_t.parameters():
            p.requires_grad_(False)

        self.class_id = class_id
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.double_q = double_q
        self.device = device  # agent/mixer/buffer are placed on device by the trainer
        # Single optimization group: agent (encoder+comm+q_head) + mixer.
        self.params = list(agent.parameters_group()) + list(mixer.parameters())
        self.opt = torch.optim.Adam(self.params, lr=lr)

    def update(self, batch) -> dict:
        cid = self.class_id
        # ---- live chosen-action Q -> live mixer (raw state) ----
        q = self.agent.q_values(batch["obs"], batch["comm_mask"], cid)         # [B,N,A]
        chosen = q.gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)      # [B,N]
        q_tot = self.mixer(chosen, batch["state"])                            # [B,1]

        # ---- bootstrap target (no_grad: target nets only, no gradient leak) ----
        with torch.no_grad():
            # value ALWAYS from the TARGET agent net (this *is* the target network).
            q_next_t = self.agent_t.q_values(batch["next_obs"], batch["next_comm_mask"], cid)
            if self.double_q:
                # action SELECTED by the LIVE net, VALUE gathered from the TARGET net.
                q_next_live = self.agent.q_values(batch["next_obs"], batch["next_comm_mask"], cid)
                next_a = q_next_live.argmax(-1, keepdim=True)                 # [B,N,1]
                next_max = q_next_t.gather(-1, next_a).squeeze(-1)            # [B,N]
            else:
                next_max = q_next_t.max(-1).values                           # [B,N]

            q_tot_next = self.mixer_t(next_max, batch["next_state"])         # [B,1]
            # mode A: `terminated` is true-termination only (~always 0 here) -> always
            # bootstrap; truncation handled at collection via TERMINAL (pre-reset)
            # next_obs/next_state. shapes: reward/terminated/q_tot_next all [B,1].
            y = batch["reward"] + self.gamma * (1.0 - batch["terminated"]) * q_tot_next

        loss = ((q_tot - y) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(self.params, self.grad_clip).item()
        self.opt.step()

        return {"td_loss": loss.item(), "grad_norm": gnorm,
                "q_tot_mean": q_tot.mean().item(), "target_mean": y.mean().item(),
                "mixer_w_min": self.mixer.weight_min(batch["state"])}

    def sync_targets(self):
        self.agent_t.load_state_dict(self.agent.state_dict())
        self.mixer_t.load_state_dict(self.mixer.state_dict())
