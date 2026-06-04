"""
Observation predictability probes.

Probes measure how much information about the target position is recoverable
from an agent's local observation.  A high R² score means the agent can predict
the target position from its observations; a low score means the observations are
effectively independent of the target (as at alpha=0 for sensor-limited agents).

Usage:
    from probes.predictability import train_probe, probe_score, probe_gap
"""

from probes.predictability import train_probe, probe_score, probe_gap, TargetPredictor

__all__ = ["TargetPredictor", "train_probe", "probe_score", "probe_gap"]
