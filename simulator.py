"""WellSimulator -- a plant stand-in for closed-loop testing.

This is a drop-in surrogate for the official event simulator. It exposes the
exact same interface the brief specifies:

    Q, WHP, FLP, BHP = simulator.step(choke_position)

so swapping in the real simulator is a one-line change in the scenario runner.

Internally it advances the four identified ARX(1) models by one control
interval and adds measurement noise at the level fitted from the step-test
residuals. It clips the commanded choke to the physical [0, 100] range (the
plant cannot exceed that); the ramp limit is an actuator/controller constraint
and is intentionally NOT enforced here, so the simulator faithfully reports what
would happen if a controller ever commanded an aggressive move.
"""
from __future__ import annotations

import numpy as np

from dataio import OUTPUTS
from model_id import ARXModel


class WellSimulator:
    """First-order well model with the official simulator's ``step`` signature."""

    def __init__(
        self,
        models: dict[str, ARXModel],
        u0: float,
        y0: dict[str, float] | None = None,
        noise: bool = True,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        models : identified ARX(1) model per output (Q, WHP, FLP, BHP).
        u0 : initial choke position (%).
        y0 : optional explicit initial outputs. If omitted, each output starts
            at the model's steady state for u0 (a well already lined out at u0).
        noise : add fitted measurement noise to each reported output.
        seed : RNG seed for reproducibility.
        """
        missing = [o for o in OUTPUTS if o not in models]
        if missing:
            raise ValueError(f"WellSimulator missing models for outputs: {missing}")

        self.models = models
        self.noise = noise
        self._rng = np.random.default_rng(seed)

        self.u = float(u0)
        if y0 is None:
            self.state = {o: models[o].steady_state(u0) for o in OUTPUTS}
        else:
            missing_y = [o for o in OUTPUTS if o not in y0]
            if missing_y:
                raise ValueError(f"y0 is missing initial values for: {missing_y}")
            self.state = {o: float(y0[o]) for o in OUTPUTS}

    def measure(self) -> tuple[float, float, float, float]:
        """Return the current (noise-free) outputs as (Q, WHP, FLP, BHP)."""
        return tuple(self.state[o] for o in OUTPUTS)  # type: ignore[return-value]

    def step(self, choke_position: float) -> tuple[float, float, float, float]:
        """Apply a choke position for one interval; return (Q, WHP, FLP, BHP).

        The commanded choke is clipped to the physical [0, 100] range. The
        internal (true) state advances noise-free; the returned measurement has
        fitted noise added on top, mirroring a real sensor reading.
        """
        u = float(min(100.0, max(0.0, choke_position)))
        self.u = u

        reported: list[float] = []
        for o in OUTPUTS:
            m = self.models[o]
            self.state[o] = m.predict_next(self.state[o], u)
            value = self.state[o]
            if self.noise and m.resid_std > 0:
                value = value + self._rng.normal(0.0, m.resid_std)
            reported.append(float(value))

        return tuple(reported)  # type: ignore[return-value]


# Startup state used by the demonstration scenarios: the well lined out at the
# low choke setting from the top of the step-test dataset (choke 30%, Q ~ 90).
STARTUP_STATE = {"Q": 90.0, "WHP": 250.0, "FLP": 180.0, "BHP": 3000.0}
STARTUP_CHOKE = 30.0


if __name__ == "__main__":
    from model_id import identify_all
    from dataio import load_step_test

    mdls = identify_all(load_step_test())
    sim = WellSimulator(mdls, u0=STARTUP_CHOKE, y0=STARTUP_STATE, noise=True, seed=0)
    print("Step response to a fixed 50% choke from startup:")
    print(f"{'k':>3}{'choke':>7}{'Q':>9}{'WHP':>9}{'FLP':>9}{'BHP':>10}")
    for k in range(0, 16):
        q, whp, flp, bhp = sim.step(50.0)
        if k % 3 == 0:
            print(f"{k:>3}{50.0:>7.1f}{q:>9.2f}{whp:>9.2f}{flp:>9.2f}{bhp:>10.2f}")
