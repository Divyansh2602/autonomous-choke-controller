"""ChokeMPC -- a simplified, safety-first predictive choke controller.

The brief explicitly permits a brute-force / candidate-search MPC. Each control
interval the controller:

  1. Reads the current measurements (Q, WHP, FLP, BHP) and choke position.
  2. Enumerates candidate next positions within the ramp limit
     (choke +/- up to max_ramp, on a fine grid), clipped to [0, 100].
  3. For each candidate, predicts the pressure trajectory over a settling
     horizon with the internal ARX(1) models AND the eventual steady state, and
     REJECTS the candidate if any pressure would fall below its lower bound plus
     a safety margin -- at any predicted step or at steady state.
  4. Among the safe (feasible) candidates, minimises
         cost = (Q_ss(candidate) - target)^2 + lambda * (delta_choke)^2
     which tracks the target when it is reachable and, because every safe
     candidate then has Q_ss < target, automatically settles at the highest
     safe rate when the target is infeasible.
  5. If NO candidate is safe (e.g. the plant is already outside the envelope),
     it falls back to the move that best restores the envelope -- backing the
     choke off to raise the pressures.

Because a first-order response moves monotonically from the current output to
its steady state, guaranteeing steady-state feasibility (plus a margin sized to
the process noise) is enough to guarantee the whole trajectory stays inside the
envelope. That is what makes this controller safe-by-construction: it never
commands a move it has not proven safe, and it knows when to say no.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dataio import OUTPUTS
from envelope import PRESSURE_OUTPUTS, DEFAULT_ENVELOPE, Envelope
from model_id import ARXModel


@dataclass
class ControlDecision:
    """Diagnostics for a single control step (for logging / plotting)."""

    choke: float
    feasible: bool           # was at least one candidate safe?
    reached_target: bool     # is the chosen steady-state Q within tol of target?
    predicted_Q_ss: float    # steady-state Q the chosen choke settles to
    n_candidates: int
    n_feasible: int


@dataclass
class ChokeMPC:
    """Constrained predictive choke controller.

    Parameters
    ----------
    models : identified ARX(1) model per output.
    envelope : operating envelope (constraints). Defaults to DEFAULT_ENVELOPE.
    horizon : prediction horizon in steps for the safety check.
    grid_step : choke resolution (%) for the candidate search.
    move_penalty : lambda weighting on (delta_choke)^2 in the cost.
    safety_sigma : safety margin is safety_sigma * process-noise-std per pressure.
    safety_floor : minimum safety margin (psi) regardless of noise.
    target_tol : |Q_ss - target| below this counts as "target reached".
    """

    models: dict[str, ARXModel]
    envelope: Envelope = DEFAULT_ENVELOPE
    horizon: int = 15
    grid_step: float = 0.5
    move_penalty: float = 1e-3
    safety_sigma: float = 4.0
    safety_floor: float = 2.0
    target_tol: float = 1.0
    last_info: ControlDecision | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        missing = [o for o in OUTPUTS if o not in self.models]
        if missing:
            raise ValueError(f"ChokeMPC missing models for outputs: {missing}")
        # Per-pressure safety margins, sized to the fitted process noise but
        # never below the floor.
        self.margins: dict[str, float] = {
            p: max(self.safety_floor, self.safety_sigma * self.models[p].resid_std)
            for p in PRESSURE_OUTPUTS
        }

    # -- internals ---------------------------------------------------------
    def _candidates(self, u_current: float) -> np.ndarray:
        """Ramp- and bound-feasible candidate choke positions, incl. current."""
        env = self.envelope
        lo = env.clip_choke(u_current - env.max_ramp)
        hi = env.clip_choke(u_current + env.max_ramp)
        grid = np.arange(lo, hi + 1e-9, self.grid_step)
        grid = np.clip(grid, env.choke_min, env.choke_max)
        # Always include the exact current position (hold) as an option.
        grid = np.unique(np.append(grid, env.clip_choke(u_current)))
        return grid

    def _is_safe(self, u_cand: float, state: dict[str, float]) -> bool:
        """True if holding u_cand keeps every pressure above its bound+margin.

        Checks the predicted transient over the horizon and the steady state.
        """
        mins = self.envelope.pressure_mins()
        for p in PRESSURE_OUTPUTS:
            m = self.models[p]
            limit = mins[p] + self.margins[p]
            # Steady state is the eventual (and, for a monotonic first-order
            # move, the worst) value when opening the choke.
            if m.steady_state(u_cand) < limit:
                return False
            # Transient trajectory from the current measured state.
            y = state[p]
            for _ in range(self.horizon):
                y = m.predict_next(y, u_cand)
                if y < limit:
                    return False
        return True

    # -- public API --------------------------------------------------------
    def decide(
        self,
        q: float,
        whp: float,
        flp: float,
        bhp: float,
        u_current: float,
        target: float,
    ) -> float:
        """Return the next choke position (%) given the current measurements."""
        state = {"Q": q, "WHP": whp, "FLP": flp, "BHP": bhp}
        candidates = self._candidates(u_current)

        q_model = self.models["Q"]
        feasible: list[tuple[float, float, float]] = []  # (cost, u, Q_ss)
        for u_cand in candidates:
            if not self._is_safe(u_cand, state):
                continue
            q_ss = q_model.steady_state(u_cand)
            cost = (q_ss - target) ** 2 + self.move_penalty * (u_cand - u_current) ** 2
            feasible.append((cost, float(u_cand), float(q_ss)))

        if feasible:
            cost, u_next, q_ss = min(feasible, key=lambda t: t[0])
            reached = abs(q_ss - target) <= self.target_tol
            self.last_info = ControlDecision(
                choke=u_next,
                feasible=True,
                reached_target=reached,
                predicted_Q_ss=q_ss,
                n_candidates=len(candidates),
                n_feasible=len(feasible),
            )
            return u_next

        # No safe candidate: back off toward the move that best restores the
        # envelope (maximise the worst steady-state pressure margin).
        def recovery_margin(u_cand: float) -> float:
            return min(
                self.models[p].steady_state(u_cand) - self.envelope.pressure_mins()[p]
                for p in PRESSURE_OUTPUTS
            )

        u_next = float(max(candidates, key=recovery_margin))
        self.last_info = ControlDecision(
            choke=u_next,
            feasible=False,
            reached_target=False,
            predicted_Q_ss=q_model.steady_state(u_next),
            n_candidates=len(candidates),
            n_feasible=0,
        )
        return u_next


@dataclass
class NaiveController:
    """Baseline: proportional Q-tracking with NO envelope awareness.

    Respects only the physical choke bounds and the ramp limit -- it chases the
    oil-rate target regardless of pressure. Used to quantify how many envelope
    violations the safety-first MPC prevents.
    """

    envelope: Envelope = DEFAULT_ENVELOPE
    gain: float = 0.5  # % choke per (bbl/hr) of oil-rate error

    def decide(
        self,
        q: float,
        whp: float,
        flp: float,
        bhp: float,
        u_current: float,
        target: float,
    ) -> float:
        """Return the next choke position, ignoring the pressure envelope."""
        delta = self.gain * (target - q)
        return self.envelope.apply_ramp(u_current + delta, u_current)


if __name__ == "__main__":
    from model_id import identify_all
    from dataio import load_step_test
    from simulator import STARTUP_CHOKE, STARTUP_STATE, WellSimulator

    mdls = identify_all(load_step_test())
    ctrl = ChokeMPC(mdls)
    print("Safety margins (psi):", {k: round(v, 2) for k, v in ctrl.margins.items()})

    sim = WellSimulator(mdls, u0=STARTUP_CHOKE, y0=STARTUP_STATE, noise=True, seed=1)
    u = STARTUP_CHOKE
    q, whp, flp, bhp = sim.measure()
    print("\nInfeasible-target demo (target=185, max safe ~156):")
    for k in range(40):
        u = ctrl.decide(q, whp, flp, bhp, u, target=185.0)
        q, whp, flp, bhp = sim.step(u)
    info = ctrl.last_info
    print(f"  settled choke={u:.1f}%  Q={q:.1f}  WHP={whp:.1f}  feasible={info.feasible}")
