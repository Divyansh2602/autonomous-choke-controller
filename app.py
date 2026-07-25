"""Live Streamlit dashboard for the autonomous choke controller.

Runs the safety-first MPC closed-loop against the WellSimulator, reusing the
exact modules the submission is built from (no duplicated controller or plant
logic), and streams the trends + per-step controller diagnostics live.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import dataclasses
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from controller import ChokeMPC
from dataio import OUTPUTS, infer_sample_time, load_step_test
from envelope import DEFAULT_ENVELOPE as ENV
from model_id import identify_all
from simulator import STARTUP_CHOKE, STARTUP_STATE, WellSimulator

st.set_page_config(page_title="Choke Controller — Problem 3A", layout="wide")


@st.cache_resource
def get_models():
    df = load_step_test()
    return identify_all(df, Ts=infer_sample_time(df))


MODELS = get_models()
MAX_SAFE_Q = MODELS["Q"].steady_state(
    (MODELS["WHP"].b0 - ENV.WHP_min) / (-MODELS["WHP"].b1)
)

# ---------------------------------------------------------------- sidebar --
st.sidebar.title("Controller demo")
st.sidebar.caption(
    "Honeywell Hackathon — Problem 3A. The controller sets the choke every "
    "hour to track the target oil rate while never breaching the pressure "
    "envelope (ramp limit ±5 %/step)."
)

target_custom = st.sidebar.slider(
    "Custom target oil rate (bbl/hr)", 90, 200, 130, 5,
    help=f"Hard max safe rate is ~{MAX_SAFE_Q:.0f} bbl/hr — ask for more and "
         "the controller settles there instead.",
)
noise_mult = st.sidebar.select_slider(
    "Measurement noise (× identified level)", options=[0, 1, 2, 4], value=1
)

st.sidebar.markdown("**Scenario presets**")
col_a, col_b, col_c = st.sidebar.columns(3)
run_a = col_a.button("A", help="Startup → 130 bbl/hr", use_container_width=True)
run_b = col_b.button("B", help="Track 100 → 150 bbl/hr", use_container_width=True)
run_c = col_c.button("C", help="Infeasible 185 bbl/hr", use_container_width=True)
run_custom = st.sidebar.button("Run custom target", use_container_width=True)
reset = st.sidebar.button("Reset", type="secondary", use_container_width=True)

if reset:
    st.session_state.pop("history", None)

# ------------------------------------------------------------ page header --
st.title("Autonomous Production Choke Controller")
banner = st.empty()
m1, m2, m3, m4 = st.columns(4)
metric_cands = m1.empty()
metric_rej = m2.empty()
metric_dchoke = m3.empty()
metric_margin = m4.empty()
chart = st.empty()
summary = st.empty()


def plant_models(mult: float):
    """Plant copy of the identified models with scaled measurement noise."""
    return {
        name: dataclasses.replace(m, resid_std=m.resid_std * mult)
        for name, m in MODELS.items()
    }


def draw(history: dict, target_label: str):
    """Render the 5-panel live trend figure from the recorded history."""
    t = np.arange(len(history["Q"]))
    fig, axes = plt.subplots(5, 1, figsize=(11, 9), sharex=True)

    ax = axes[0]
    ax.plot(t, history["target"], "b--", lw=1.6, label="Target oil rate")
    ax.plot(t, history["Q"], color="tab:green", lw=1.7, label="Actual oil rate")
    ax.set_ylabel("Q (bbl/hr)")
    ax.legend(fontsize=8, loc="lower right")

    mins = ENV.pressure_mins()
    colors = {"WHP": "tab:red", "FLP": "tab:orange", "BHP": "tab:purple"}
    for ax, name in zip(axes[1:4], ("WHP", "FLP", "BHP")):
        ax.plot(t, history[name], color=colors[name], lw=1.5, label=name)
        ax.axhline(mins[name], color="red", ls="--", lw=1.4,
                   label=f"{name} lower bound = {mins[name]:g} psi")
        ax.set_ylabel(f"{name} (psi)")
        ax.legend(fontsize=8, loc="upper right")

    ax = axes[4]
    ax.plot(t, history["u"], color="tab:brown", lw=1.7, label="Choke position")
    ax.set_ylabel("Choke (%)")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Time (hr)")
    ax.annotate(f"ramp limit ±{ENV.max_ramp:g} %/step", xy=(0.99, 0.9),
                xycoords="axes fraction", ha="right", fontsize=9, color="0.35")
    ax.legend(fontsize=8, loc="lower right")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(target_label, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def run_scenario(label: str, target_fn, n_steps: int):
    """Run one closed-loop scenario live, updating the page every step."""
    sim = WellSimulator(plant_models(noise_mult), u0=STARTUP_CHOKE,
                        y0=STARTUP_STATE, noise=noise_mult > 0, seed=0)
    ctrl = ChokeMPC(MODELS, envelope=ENV)

    hist = {k: [] for k in ["target", "Q", "WHP", "FLP", "BHP", "u"]}
    u = float(STARTUP_CHOKE)
    q, whp, flp, bhp = sim.measure()
    violations = 0

    for k in range(n_steps):
        target = float(target_fn(k))
        hist["target"].append(target)
        hist["Q"].append(q); hist["WHP"].append(whp)
        hist["FLP"].append(flp); hist["BHP"].append(bhp)
        hist["u"].append(u)

        if ENV.any_violation(whp, flp, bhp):
            violations += 1

        u_prev = u
        u = ctrl.decide(q, whp, flp, bhp, u, target)
        info = ctrl.last_info

        # Per-step controller diagnostics.
        metric_cands.metric("Candidates evaluated", info.n_candidates)
        metric_rej.metric("Rejected as unsafe", info.n_candidates - info.n_feasible)
        metric_dchoke.metric("Chosen Δchoke", f"{u - u_prev:+.1f} %")
        metric_margin.metric("WHP margin", f"{whp - ENV.WHP_min:.1f} psi")

        if info.reached_target:
            banner.success(f"🟢 Tracking target {target:.0f} bbl/hr — inside envelope")
        elif info.feasible:
            banner.warning(
                f"🟠 Target {target:.0f} bbl/hr infeasible — settling at max "
                f"achievable safe rate (~{info.predicted_Q_ss:.0f} bbl/hr)"
            )

        if k % 2 == 0 or k == n_steps - 1:
            fig = draw(hist, label)
            chart.pyplot(fig)
            plt.close(fig)
        time.sleep(0.02)

        q, whp, flp, bhp = sim.step(u)

    # The red state must be unreachable: fail loudly if it ever is not.
    assert violations == 0, f"envelope violated {violations}× — must be impossible"

    st.session_state["history"] = hist
    st.session_state["label"] = label
    settled = float(np.mean(hist["Q"][-10:]))
    summary.info(
        f"**Run complete — {violations} envelope violations.** "
        f"Settled at {settled:.1f} bbl/hr (final target "
        f"{hist['target'][-1]:.0f}), choke {hist['u'][-1]:.1f} %."
    )


if run_a:
    run_scenario("Scenario A — Startup to 130 bbl/hr", lambda k: 130.0, 60)
elif run_b:
    run_scenario("Scenario B — Target 100 → 150 bbl/hr",
                 lambda k: 100.0 if k < 50 else 150.0, 100)
elif run_c:
    run_scenario("Scenario C — Infeasible 185 bbl/hr", lambda k: 185.0, 70)
elif run_custom:
    run_scenario(f"Custom target {target_custom} bbl/hr",
                 lambda k: float(target_custom), 60)
elif "history" in st.session_state:
    fig = draw(st.session_state["history"], st.session_state["label"])
    chart.pyplot(fig)
    plt.close(fig)
    banner.info("Previous run shown — pick a scenario to run again.")
else:
    banner.info(
        "Pick a scenario in the sidebar. The controller only ever commands "
        "moves it has predicted safe — watch the 'rejected as unsafe' counter "
        "when the well approaches the WHP floor."
    )
