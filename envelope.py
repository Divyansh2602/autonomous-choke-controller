"""Operating envelope and actuator limits -- the single source of truth.

The hackathon brief does not give explicit envelope numbers, so these are a
DOCUMENTED ASSUMPTION (see HANDOFF.md Section 3). They are chosen so that the
maximum safe production rate sits high in the observed range and Scenario C is
clearly infeasible, with WHP as the first binding constraint.

When the official envelope / simulator is announced at the event, edit ONLY this
file -- every other module reads its limits from here.
"""
from __future__ import annotations

from dataclasses import dataclass


# The pressure outputs that carry a safety constraint. All three are LOWER
# bounds: opening the choke raises oil rate but pulls every pressure down, so
# the binding constraints are floors, not ceilings (see HANDOFF.md Section 2).
PRESSURE_OUTPUTS = ("WHP", "FLP", "BHP")


@dataclass(frozen=True)
class Envelope:
    """Immutable container for all operating constraints.

    Attributes
    ----------
    WHP_min, FLP_min, BHP_min : float
        Lower pressure bounds (psi). Violating any of these is a hard safety
        breach.
    choke_min, choke_max : float
        Absolute choke position limits (%).
    max_ramp : float
        Maximum change in choke position per control interval (% per step).
    Ts_hr : float
        Control interval (hr).
    """

    WHP_min: float = 215.0
    FLP_min: float = 150.0
    BHP_min: float = 2850.0
    choke_min: float = 0.0
    choke_max: float = 100.0
    max_ramp: float = 5.0
    Ts_hr: float = 1.0

    def pressure_mins(self) -> dict[str, float]:
        """Return the lower bound for each constrained pressure output."""
        return {"WHP": self.WHP_min, "FLP": self.FLP_min, "BHP": self.BHP_min}

    def clip_choke(self, u: float) -> float:
        """Clip a choke position to the absolute [choke_min, choke_max] range."""
        return float(min(self.choke_max, max(self.choke_min, u)))

    def apply_ramp(self, u_target: float, u_current: float) -> float:
        """Clamp a desired choke move to the ramp limit, then to absolute bounds.

        Guarantees ``|return - u_current| <= max_ramp`` and the result lies in
        ``[choke_min, choke_max]``.
        """
        delta = u_target - u_current
        if delta > self.max_ramp:
            delta = self.max_ramp
        elif delta < -self.max_ramp:
            delta = -self.max_ramp
        return self.clip_choke(u_current + delta)

    def pressure_violations(self, whp: float, flp: float, bhp: float) -> dict[str, bool]:
        """Return, per pressure, whether it is strictly below its lower bound."""
        return {
            "WHP": whp < self.WHP_min,
            "FLP": flp < self.FLP_min,
            "BHP": bhp < self.BHP_min,
        }

    def any_violation(self, whp: float, flp: float, bhp: float) -> bool:
        """True if any constrained pressure is below its lower bound."""
        return any(self.pressure_violations(whp, flp, bhp).values())

    def worst_margin(self, whp: float, flp: float, bhp: float) -> float:
        """Smallest (pressure - lower_bound) across the three pressures (psi).

        Positive means all pressures are inside the envelope; the value is the
        tightest remaining headroom. Negative means at least one bound is
        breached, by that amount. Used as the fallback objective when no
        candidate move is safe -- maximise the worst margin to climb back in.
        """
        return min(
            whp - self.WHP_min,
            flp - self.FLP_min,
            bhp - self.BHP_min,
        )


# Default envelope instance used across the project.
DEFAULT_ENVELOPE = Envelope()


if __name__ == "__main__":
    env = DEFAULT_ENVELOPE
    print("Operating envelope (single source of truth):")
    print(f"  WHP_min = {env.WHP_min} psi")
    print(f"  FLP_min = {env.FLP_min} psi")
    print(f"  BHP_min = {env.BHP_min} psi")
    print(f"  choke   in [{env.choke_min}, {env.choke_max}] %")
    print(f"  ramp    <= {env.max_ramp} % per {env.Ts_hr} hr step")
