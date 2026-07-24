"""Closed-loop scenario runners, metrics, and the naive-baseline comparison.

Three demonstration scenarios from the brief:

  A - Startup -> Target: bring the well from startup to a fixed production target.
  B - Target Tracking:   target steps mid-run (100 -> 150 bbl/hr).
  C - Infeasible Target: request exceeds what is safe -> settle at the max safe rate.

Each scenario is run closed-loop against the WellSimulator and summarised with
metrics (constraint violations, tracking error, settled rate). The same
scenarios are also run with a naive, envelope-unaware baseline to quantify how
many violations the safety-first MPC prevents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from controller import ChokeMPC, NaiveController
from dataio import OUTPUTS
from envelope import DEFAULT_ENVELOPE, PRESSURE_OUTPUTS, Envelope
from model_id import ARXModel
from simulator import STARTUP_CHOKE, STARTUP_STATE, WellSimulator


@dataclass
class ScenarioResult:
    """Time series and metrics for one closed-loop run."""

    name: str
    title: str
    t: np.ndarray
    target: np.ndarray
    Q: np.ndarray
    WHP: np.ndarray
    FLP: np.ndarray
    BHP: np.ndarray
    u: np.ndarray
    metrics: dict = field(default_factory=dict)


@dataclass
class ScenarioSpec:
    """Definition of a scenario: how long, what target, where it starts."""

    name: str
    title: str
    n_steps: int
    target_fn: Callable[[int], float]
    u0: float = STARTUP_CHOKE
    y0: dict[str, float] | None = None
    feasible: bool = True  # is the (final) target physically achievable safely?

    def target_series(self) -> np.ndarray:
        return np.array([self.target_fn(k) for k in range(self.n_steps)], dtype=float)


def run_closed_loop(
    controller,
    models: dict[str, ARXModel],
    spec: ScenarioSpec,
    envelope: Envelope = DEFAULT_ENVELOPE,
    noise: bool = True,
    seed: int = 0,
) -> ScenarioResult:
    """Run one scenario closed-loop and return the recorded time series.

    Causal loop: at each step we record the current measurement and the choke
    that produced it, ask the controller for the next choke, then advance the
    plant one interval.
    """
    targets = spec.target_series()
    n = spec.n_steps
    y0 = spec.y0 if spec.y0 is not None else STARTUP_STATE
    sim = WellSimulator(models, u0=spec.u0, y0=y0, noise=noise, seed=seed)

    t = np.arange(n, dtype=float) * envelope.Ts_hr
    Q = np.empty(n); WHP = np.empty(n); FLP = np.empty(n); BHP = np.empty(n)
    u_rec = np.empty(n)

    u = float(spec.u0)
    q, whp, flp, bhp = sim.measure()
    for k in range(n):
        Q[k], WHP[k], FLP[k], BHP[k], u_rec[k] = q, whp, flp, bhp, u
        u = controller.decide(q, whp, flp, bhp, u, targets[k])
        q, whp, flp, bhp = sim.step(u)

    result = ScenarioResult(
        name=spec.name, title=spec.title, t=t, target=targets,
        Q=Q, WHP=WHP, FLP=FLP, BHP=BHP, u=u_rec,
    )
    result.metrics = compute_metrics(result, envelope, spec)
    return result


def compute_metrics(
    result: ScenarioResult, envelope: Envelope, spec: ScenarioSpec
) -> dict:
    """Summarise a run: violations, tracking error, settled rate."""
    mins = envelope.pressure_mins()
    series = {"WHP": result.WHP, "FLP": result.FLP, "BHP": result.BHP}

    violations_by_pressure = {
        p: int(np.sum(series[p] < mins[p])) for p in PRESSURE_OUTPUTS
    }
    violations_total = int(sum(violations_by_pressure.values()))

    min_pressures = {p: float(np.min(series[p])) for p in PRESSURE_OUTPUTS}
    worst_margin = float(
        min(min_pressures[p] - mins[p] for p in PRESSURE_OUTPUTS)
    )

    # Tracking assessed over the last 10 steps (settled window).
    tail = slice(-10, None)
    final_target = float(result.target[-1])
    final_Q = float(np.mean(result.Q[tail]))
    steady_state_error = final_Q - final_target
    max_abs_tracking_error = float(np.max(np.abs(result.Q - result.target)))

    return {
        "scenario": result.name,
        "target_feasible": spec.feasible,
        "final_target_bbl_hr": final_target,
        "settled_oil_rate_bbl_hr": round(final_Q, 2),
        "settled_choke_pct": round(float(np.mean(result.u[tail])), 2),
        "steady_state_error_bbl_hr": round(steady_state_error, 2),
        "max_abs_tracking_error_bbl_hr": round(max_abs_tracking_error, 2),
        "violations_total": violations_total,
        "violations_by_pressure": violations_by_pressure,
        "min_pressures_psi": {p: round(v, 2) for p, v in min_pressures.items()},
        "worst_pressure_margin_psi": round(worst_margin, 2),
    }


# -- Scenario catalogue ---------------------------------------------------
def scenario_specs() -> list[ScenarioSpec]:
    """The three required demonstration scenarios (targets per HANDOFF Section 3)."""
    return [
        ScenarioSpec(
            name="A",
            title="Scenario A -- Startup to Target (130 bbl/hr)",
            n_steps=60,
            target_fn=lambda k: 130.0,
            feasible=True,
        ),
        ScenarioSpec(
            name="B",
            title="Scenario B -- Target Tracking (100 then 150 bbl/hr)",
            n_steps=100,
            target_fn=lambda k: 100.0 if k < 50 else 150.0,
            feasible=True,
        ),
        ScenarioSpec(
            name="C",
            title="Scenario C -- Infeasible Target (185 bbl/hr, settle at max safe)",
            n_steps=70,
            target_fn=lambda k: 185.0,
            feasible=False,
        ),
    ]


def run_all_mpc(
    models: dict[str, ARXModel],
    envelope: Envelope = DEFAULT_ENVELOPE,
    seed: int = 0,
) -> dict[str, ScenarioResult]:
    """Run all three scenarios with the safety-first MPC."""
    results: dict[str, ScenarioResult] = {}
    for spec in scenario_specs():
        ctrl = ChokeMPC(models, envelope=envelope)
        results[spec.name] = run_closed_loop(ctrl, models, spec, envelope, seed=seed)
    return results


def run_all_baseline(
    models: dict[str, ARXModel],
    envelope: Envelope = DEFAULT_ENVELOPE,
    seed: int = 0,
) -> dict[str, ScenarioResult]:
    """Run all three scenarios with the naive (envelope-unaware) baseline."""
    results: dict[str, ScenarioResult] = {}
    for spec in scenario_specs():
        ctrl = NaiveController(envelope=envelope)
        results[spec.name] = run_closed_loop(ctrl, models, spec, envelope, seed=seed)
    return results


if __name__ == "__main__":
    from dataio import load_step_test
    from model_id import identify_all

    mdls = identify_all(load_step_test())
    mpc = run_all_mpc(mdls)
    naive = run_all_baseline(mdls)
    print(f"{'scen':<5}{'MPC viol':>10}{'naive viol':>12}{'settled Q':>12}{'ss err':>9}")
    for name in ("A", "B", "C"):
        m = mpc[name].metrics
        b = naive[name].metrics
        print(
            f"{name:<5}{m['violations_total']:>10}{b['violations_total']:>12}"
            f"{m['settled_oil_rate_bbl_hr']:>12}{m['steady_state_error_bbl_hr']:>9}"
        )
