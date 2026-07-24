"""ARX(1) dynamic model identification from the open-loop step test.

For each output y in {Q, WHP, FLP, BHP} we fit a first-order ARX model that
maps the choke position u to the output one step ahead:

    y[k+1] = a * y[k] + (1 - a) * (b0 + b1 * u[k])

This is linear in the parameters, so we regress y[k+1] on [y[k], 1, u[k]] by
ordinary least squares to recover (a, c0, c1), then map back to the physically
meaningful form:

    b1 = c1 / (1 - a)        steady-state gain     dy_ss/du
    b0 = c0 / (1 - a)        steady-state intercept
    tau = -Ts / ln(a)        first-order time constant (hr)

The one-step residual standard deviation gives the process noise level, which
the simulator reuses so its measurements have realistic scatter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataio import OUTPUTS, to_canonical


@dataclass(frozen=True)
class ARXModel:
    """Identified first-order ARX(1) model for one output.

    ``y[k+1] = a * y[k] + (1 - a) * (b0 + b1 * u[k])``
    """

    name: str
    a: float          # pole (0 < a < 1 for a stable first-order response)
    b0: float         # steady-state intercept
    b1: float         # steady-state gain dy_ss/du
    resid_std: float  # one-step-ahead residual std (process noise level)
    Ts_hr: float      # sample interval used for the time-constant conversion

    @property
    def gain(self) -> float:
        """Steady-state gain dy_ss/du (== b1)."""
        return self.b1

    @property
    def tau_hr(self) -> float:
        """First-order time constant (hr). Infinite if a >= 1 (non-decaying)."""
        if not (0.0 < self.a < 1.0):
            return float("inf")
        return -self.Ts_hr / np.log(self.a)

    def predict_next(self, y: float, u: float) -> float:
        """One-step-ahead prediction y[k+1] from current output y and input u."""
        return self.a * y + (1.0 - self.a) * (self.b0 + self.b1 * u)

    def steady_state(self, u: float) -> float:
        """Steady-state output the model settles to when the choke is held at u."""
        return self.b0 + self.b1 * u


def identify_arx(y: np.ndarray, u: np.ndarray, name: str, Ts: float = 1.0) -> ARXModel:
    """Identify a single ARX(1) model by OLS.

    Parameters
    ----------
    y : array of output samples y[0..N-1].
    u : array of input samples  u[0..N-1] (choke %).
    name : output label.
    Ts : sample interval (hr).
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    if y.shape != u.shape or y.ndim != 1:
        raise ValueError("y and u must be 1-D arrays of equal length.")
    if len(y) < 4:
        raise ValueError(f"Need at least 4 samples to identify '{name}', got {len(y)}.")

    # Regress y[k+1] on [y[k], 1, u[k]].
    y_k = y[:-1]
    y_k1 = y[1:]
    u_k = u[:-1]
    X = np.column_stack([y_k, np.ones_like(y_k), u_k])

    theta, *_ = np.linalg.lstsq(X, y_k1, rcond=None)
    a, c0, c1 = (float(v) for v in theta)

    if a >= 1.0:
        raise ValueError(
            f"Identified pole a={a:.4f} >= 1 for '{name}': model is not a stable "
            "first-order response; check the data."
        )

    b1 = c1 / (1.0 - a)
    b0 = c0 / (1.0 - a)

    resid = y_k1 - X @ theta
    # dof correction: N-1 equations minus 3 fitted parameters.
    dof = max(1, len(resid) - 3)
    resid_std = float(np.sqrt(np.sum(resid**2) / dof))

    return ARXModel(name=name, a=a, b0=b0, b1=b1, resid_std=resid_std, Ts_hr=Ts)


def identify_all(df: pd.DataFrame, Ts: float = 1.0) -> dict[str, ARXModel]:
    """Identify an ARX(1) model for every output in a tidy (canonical) frame.

    Accepts either raw or canonical column names.
    """
    canon = df if "u" in df.columns else to_canonical(df)
    u = canon["u"].to_numpy(dtype=float)
    return {name: identify_arx(canon[name].to_numpy(float), u, name, Ts) for name in OUTPUTS}


def free_run(model: ARXModel, u: np.ndarray, y0: float) -> np.ndarray:
    """Simulate the model open-loop (free-run) from y0 under an input sequence.

    Uses only the model's own predictions (never the measured output), so the
    result is a genuine multi-step validation, not a one-step-ahead cheat.
    """
    u = np.asarray(u, dtype=float)
    y = np.empty(len(u), dtype=float)
    y[0] = y0
    for k in range(len(u) - 1):
        y[k + 1] = model.predict_next(y[k], u[k])
    return y


def validation_fit(model: ARXModel, df: pd.DataFrame) -> dict[str, float]:
    """Free-run the model over the dataset and report fit quality vs the data."""
    canon = df if "u" in df.columns else to_canonical(df)
    y_meas = canon[model.name].to_numpy(float)
    u = canon["u"].to_numpy(float)
    y_sim = free_run(model, u, y0=y_meas[0])

    err = y_sim - y_meas
    rmse = float(np.sqrt(np.mean(err**2)))
    span = float(np.ptp(y_meas)) or 1.0
    return {
        "rmse": rmse,
        "rmse_pct_of_span": 100.0 * rmse / span,
        "max_abs_err": float(np.max(np.abs(err))),
    }


def format_report(models: dict[str, ARXModel]) -> str:
    """Human-readable summary of the identified parameters."""
    lines = [
        "Identified ARX(1) models:  y[k+1] = a*y[k] + (1-a)*(b0 + b1*u[k])",
        "-" * 68,
        f"{'output':<6}{'a':>8}{'gain b1':>12}{'b0':>12}{'tau (hr)':>11}{'noise sd':>11}",
        "-" * 68,
    ]
    for name in OUTPUTS:
        m = models[name]
        lines.append(
            f"{m.name:<6}{m.a:>8.4f}{m.b1:>12.4f}{m.b0:>12.3f}"
            f"{m.tau_hr:>11.2f}{m.resid_std:>11.3f}"
        )
    lines.append("-" * 68)
    return "\n".join(lines)


if __name__ == "__main__":
    from dataio import infer_sample_time, load_step_test

    frame = load_step_test()
    ts = infer_sample_time(frame)
    mdls = identify_all(frame, Ts=ts)
    print(format_report(mdls))
    print("\nFree-run validation (model simulated open-loop vs data):")
    for out_name in OUTPUTS:
        v = validation_fit(mdls[out_name], frame)
        print(
            f"  {out_name:<4} RMSE={v['rmse']:.3f} "
            f"({v['rmse_pct_of_span']:.1f}% of span, max abs err {v['max_abs_err']:.2f})"
        )
