"""Plotting: the required 6-trend scenario figure and the model-validation plot.

All figures are rendered with the non-interactive Agg backend so the pipeline
runs headless. Colours and layout are kept deliberately plain and legible for a
process-control audience.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dataio import OUTPUTS, to_canonical  # noqa: E402
from envelope import Envelope  # noqa: E402
from model_id import ARXModel, free_run  # noqa: E402

# Axis labels for each output.
_YLABEL = {
    "Q": "Oil rate\n(bbl/hr)",
    "WHP": "WHP (psi)",
    "FLP": "FLP (psi)",
    "BHP": "BHP (psi)",
}


def plot_model_validation(
    models: dict[str, ARXModel], df: Any, path: str | Path
) -> Path:
    """Overlay each identified model's open-loop free-run on the measured data."""
    canon = df if "u" in df.columns else to_canonical(df)
    t = canon["t"].to_numpy(float)
    u = canon["u"].to_numpy(float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    fig.suptitle(
        "Model identification: ARX(1) free-run vs. step-test data", fontsize=13
    )

    for ax, name in zip(axes.flat, OUTPUTS):
        y_meas = canon[name].to_numpy(float)
        y_sim = free_run(models[name], u, y0=y_meas[0])
        ax.plot(t, y_meas, color="0.35", lw=1.2, label="data")
        ax.plot(t, y_sim, color="tab:red", lw=1.6, ls="--", label="ARX(1) model")
        m = models[name]
        ax.set_title(
            f"{name}:  gain={m.gain:+.2f},  tau={m.tau_hr:.1f} hr", fontsize=10
        )
        ax.set_ylabel(_YLABEL[name])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    for ax in axes[-1]:
        ax.set_xlabel("Time (hr)")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = Path(path)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_scenario(result: Any, envelope: Envelope, path: str | Path) -> Path:
    """Render the required 6-trend figure for one closed-loop scenario.

    The six trends are: Target Oil Rate, Actual Oil Rate, WHP, FLP, BHP and
    Choke Position. Oil-rate target and actual share the top panel; each
    pressure panel shows its lower-bound constraint; the choke panel shows the
    ramp context.

    ``result`` is any object exposing arrays t, target, Q, WHP, FLP, BHP, u and
    a string ``label`` / ``name`` (duck-typed to avoid a circular import).
    """
    t = np.asarray(result.t, float)

    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    title = getattr(result, "title", getattr(result, "name", "Scenario"))
    fig.suptitle(title, fontsize=13)

    # Panel 1: oil rate -- target + actual (two of the six trends).
    ax = axes[0]
    ax.plot(t, result.target, color="tab:blue", lw=1.8, ls="--", label="Target oil rate")
    ax.plot(t, result.Q, color="tab:green", lw=1.6, label="Actual oil rate")
    ax.set_ylabel(_YLABEL["Q"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # Panels 2-4: pressures with their lower-bound constraints.
    mins = envelope.pressure_mins()
    colors = {"WHP": "tab:red", "FLP": "tab:orange", "BHP": "tab:purple"}
    for ax, name in zip(axes[1:4], ("WHP", "FLP", "BHP")):
        series = np.asarray(getattr(result, name), float)
        ax.plot(t, series, color=colors[name], lw=1.5, label=name)
        ax.axhline(mins[name], color="k", ls=":", lw=1.3, label=f"{name}_min = {mins[name]:g}")
        # Shade any samples that fall below the bound (should be none for MPC).
        breach = series < mins[name]
        if breach.any():
            ax.fill_between(t, mins[name], series, where=breach, color="red", alpha=0.3)
        ax.set_ylabel(_YLABEL[name])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # Panel 5: choke position.
    ax = axes[4]
    ax.plot(t, result.u, color="tab:brown", lw=1.6, label="Choke position")
    ax.set_ylabel("Choke (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("Time (hr)")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = Path(path)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_baseline_comparison(
    mpc_result: Any, naive_result: Any, envelope: Envelope, path: str | Path
) -> Path:
    """Compare safety-first MPC vs the naive baseline on the same scenario.

    Highlights how the naive controller breaches the pressure envelope while the
    MPC stays inside it, at the cost of chasing an infeasible oil-rate target.
    """
    t = np.asarray(mpc_result.t, float)
    mins = envelope.pressure_mins()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        f"Safety-first MPC vs. naive baseline -- {getattr(mpc_result, 'title', '')}",
        fontsize=12,
    )

    ax = axes[0]
    ax.plot(t, mpc_result.target, color="tab:blue", ls="--", lw=1.6, label="Target")
    ax.plot(t, mpc_result.Q, color="tab:green", lw=1.6, label="MPC oil rate")
    ax.plot(t, naive_result.Q, color="tab:gray", lw=1.4, label="Naive oil rate")
    ax.set_ylabel(_YLABEL["Q"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t, mpc_result.WHP, color="tab:green", lw=1.6, label="MPC WHP")
    ax.plot(t, naive_result.WHP, color="tab:gray", lw=1.4, label="Naive WHP")
    ax.axhline(mins["WHP"], color="k", ls=":", lw=1.3, label=f"WHP_min = {mins['WHP']:g}")
    naive_whp = np.asarray(naive_result.WHP, float)
    breach = naive_whp < mins["WHP"]
    if breach.any():
        ax.fill_between(t, mins["WHP"], naive_whp, where=breach, color="red", alpha=0.3,
                        label="Naive breach")
    ax.set_ylabel(_YLABEL["WHP"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(t, mpc_result.u, color="tab:green", lw=1.6, label="MPC choke")
    ax.plot(t, naive_result.u, color="tab:gray", lw=1.4, label="Naive choke")
    ax.set_ylabel("Choke (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlabel("Time (hr)")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = Path(path)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
