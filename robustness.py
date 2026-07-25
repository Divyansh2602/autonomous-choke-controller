"""Robustness study: does "0 violations" survive conditions the controller
was never tuned for?

Three stress families, all run closed-loop with the controller's INTERNAL model
kept at the identified one while the TRUE plant is altered:

  1. Measurement-noise sweep -- plant noise at 1x / 2x / 4x the identified
     level, three seeds each.
  2. Plant-model mismatch -- true steady-state gains scaled +/-20 % (and a
     stress case: 20 % steeper pressure gains, 20 % weaker oil gain, 30 %
     slower dynamics). Gains are re-anchored at the startup choke so the
     startup state stays physical.
  3. Unmeasured disturbance -- a persistent -8 psi WHP shift appears mid-run
     (e.g. a backpressure change the controller is never told about).

Results go to results/robustness_metrics.json plus two plots. The numbers are
reported as measured -- nothing here is asserted into being.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from controller import ChokeMPC  # noqa: E402
from dataio import OUTPUTS, infer_sample_time, load_step_test  # noqa: E402
from envelope import DEFAULT_ENVELOPE as ENV  # noqa: E402
from envelope import PRESSURE_OUTPUTS  # noqa: E402
from model_id import ARXModel, identify_all  # noqa: E402
from scenarios import scenario_specs  # noqa: E402
from simulator import STARTUP_CHOKE, STARTUP_STATE, WellSimulator  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def perturb(
    models: dict[str, ARXModel],
    gain_scale: dict[str, float] | float = 1.0,
    tau_scale: float = 1.0,
    noise_scale: float = 1.0,
    anchor_u: float = STARTUP_CHOKE,
) -> dict[str, ARXModel]:
    """Return a perturbed copy of the models to act as the TRUE plant.

    Gains are re-anchored so the steady state at ``anchor_u`` is unchanged --
    the mismatch grows as the well moves away from the startup point, which is
    the physically meaningful kind of model error.
    """
    if isinstance(gain_scale, (int, float)):
        gain_scale = {name: float(gain_scale) for name in OUTPUTS}
    out: dict[str, ARXModel] = {}
    for name, m in models.items():
        b1 = m.b1 * gain_scale.get(name, 1.0)
        b0 = (m.b0 + m.b1 * anchor_u) - b1 * anchor_u
        a = m.a
        if 0.0 < m.a < 1.0 and tau_scale != 1.0:
            a = float(np.exp(-m.Ts_hr / (m.tau_hr * tau_scale)))
        out[name] = dataclasses.replace(
            m, a=a, b0=b0, b1=b1, resid_std=m.resid_std * noise_scale
        )
    return out


def run_case(
    ctrl_models: dict[str, ARXModel],
    plant_models: dict[str, ARXModel],
    target_fn,
    n_steps: int,
    seed: int = 0,
    disturb: tuple[int, str, float] | None = None,
) -> dict:
    """Closed loop: controller on ctrl_models, plant on plant_models.

    ``disturb`` = (step_k, output, delta_b0): at step_k the plant's output map
    shifts by delta_b0 -- persistent and unmeasured.
    """
    sim = WellSimulator(dict(plant_models), u0=STARTUP_CHOKE,
                        y0=STARTUP_STATE, noise=True, seed=seed)
    ctrl = ChokeMPC(ctrl_models, envelope=ENV)

    n = n_steps
    rec = {k: np.empty(n) for k in ("target", "Q", "WHP", "FLP", "BHP", "u")}
    u = float(STARTUP_CHOKE)
    q, whp, flp, bhp = sim.measure()
    for k in range(n):
        if disturb is not None and k == disturb[0]:
            name, delta = disturb[1], disturb[2]
            sim.models[name] = dataclasses.replace(
                sim.models[name], b0=sim.models[name].b0 + delta
            )
        rec["target"][k] = target_fn(k)
        rec["Q"][k], rec["WHP"][k], rec["FLP"][k], rec["BHP"][k] = q, whp, flp, bhp
        rec["u"][k] = u
        u = ctrl.decide(q, whp, flp, bhp, u, rec["target"][k])
        q, whp, flp, bhp = sim.step(u)

    mins = ENV.pressure_mins()
    violations = int(sum(int(np.sum(rec[p] < mins[p])) for p in PRESSURE_OUTPUTS))
    worst_margin = float(min(np.min(rec[p]) - mins[p] for p in PRESSURE_OUTPUTS))
    return {
        "violations": violations,
        "worst_margin_psi": round(worst_margin, 2),
        "settled_Q_bbl_hr": round(float(np.mean(rec["Q"][-10:])), 1),
        "settled_choke_pct": round(float(np.mean(rec["u"][-10:])), 1),
        "_rec": rec,
    }


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    df = load_step_test()
    models = identify_all(df, Ts=infer_sample_time(df))
    specs = {s.name: s for s in scenario_specs()}

    results: dict = {"noise_sweep": {}, "model_mismatch": {}, "disturbance": {}}

    # -- 1. Noise sweep ----------------------------------------------------
    print("1) Measurement-noise sweep (3 seeds each):")
    for mult in (1, 2, 4):
        for name, spec in specs.items():
            per_seed = [
                run_case(models, perturb(models, noise_scale=mult),
                         spec.target_fn, spec.n_steps, seed=s)
                for s in (0, 1, 2)
            ]
            worst = min(per_seed, key=lambda r: r["worst_margin_psi"])
            results["noise_sweep"][f"{name}_noise{mult}x"] = {
                "violations_max_over_seeds": max(r["violations"] for r in per_seed),
                "worst_margin_psi": worst["worst_margin_psi"],
                "settled_Q_bbl_hr": per_seed[0]["settled_Q_bbl_hr"],
            }
            v = results["noise_sweep"][f"{name}_noise{mult}x"]
            print(f"   {name} @ {mult}x noise: violations={v['violations_max_over_seeds']}, "
                  f"worst margin={v['worst_margin_psi']} psi")

    # 1b. The margin is a dial: at 4x noise the fixed margin (sized to the
    # identified 1x noise) can be grazed. Give the controller the TRUE noise
    # level -- its margins scale automatically (4 sigma of the real noise) and
    # safety is restored, at the cost of settling a little lower.
    print("\n1b) Same 4x-noise plant, but controller margins sized to the true noise:")
    noisy = perturb(models, noise_scale=4)
    for name, spec in specs.items():
        per_seed = [run_case(noisy, noisy, spec.target_fn, spec.n_steps, seed=s)
                    for s in (0, 1, 2)]
        worst = min(per_seed, key=lambda r: r["worst_margin_psi"])
        results["noise_sweep"][f"{name}_noise4x_margin_aware"] = {
            "violations_max_over_seeds": max(r["violations"] for r in per_seed),
            "worst_margin_psi": worst["worst_margin_psi"],
            "settled_Q_bbl_hr": per_seed[0]["settled_Q_bbl_hr"],
        }
        v = results["noise_sweep"][f"{name}_noise4x_margin_aware"]
        print(f"   {name}: violations={v['violations_max_over_seeds']}, "
              f"worst margin={v['worst_margin_psi']} psi, "
              f"settled Q={v['settled_Q_bbl_hr']}")

    # -- 2. Plant-model mismatch ------------------------------------------
    print("\n2) Plant-model mismatch (controller keeps identified model):")
    mismatch_cases = {
        "gains_-20pct": dict(gain_scale=0.8),
        "gains_+20pct": dict(gain_scale=1.2),
        "stress_steeper_slower": dict(
            gain_scale={"Q": 0.8, "WHP": 1.2, "FLP": 1.2, "BHP": 1.2},
            tau_scale=1.3,
        ),
    }
    for case, kw in mismatch_cases.items():
        for name, spec in specs.items():
            r = run_case(models, perturb(models, **kw), spec.target_fn,
                         spec.n_steps, seed=0)
            r.pop("_rec")
            results["model_mismatch"][f"{name}_{case}"] = r
            print(f"   {name} {case}: violations={r['violations']}, "
                  f"worst margin={r['worst_margin_psi']} psi, "
                  f"settled Q={r['settled_Q_bbl_hr']}")

    # -- 3. Unmeasured disturbance ----------------------------------------
    print("\n3) Unmeasured disturbance (-8 psi WHP shift at t=60, scenario B):")
    spec = specs["B"]
    rd = run_case(models, perturb(models), spec.target_fn, spec.n_steps,
                  seed=0, disturb=(60, "WHP", -8.0))
    rec = rd.pop("_rec")
    results["disturbance"]["B_whp_-8psi_at_t60"] = rd
    print(f"   violations={rd['violations']}, worst margin={rd['worst_margin_psi']} psi, "
          f"settled Q={rd['settled_Q_bbl_hr']}")

    # Disturbance time-series plot.
    t = np.arange(len(rec["WHP"]))
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    axes[0].plot(t, rec["target"], "b--", lw=1.5, label="Target")
    axes[0].plot(t, rec["Q"], color="tab:green", lw=1.5, label="Oil rate")
    axes[0].set_ylabel("Q (bbl/hr)"); axes[0].legend(fontsize=8)
    axes[1].plot(t, rec["WHP"], color="tab:red", lw=1.5, label="WHP")
    axes[1].axhline(ENV.WHP_min, color="k", ls=":", lw=1.3, label="WHP_min")
    axes[1].axvline(60, color="0.3", ls="--", lw=1.2)
    axes[1].annotate("unmeasured -8 psi shift", xy=(60, rec["WHP"].max() - 4),
                     xytext=(63, rec["WHP"].max() - 2), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="0.3"))
    axes[1].set_ylabel("WHP (psi)"); axes[1].legend(fontsize=8)
    axes[2].plot(t, rec["u"], color="tab:brown", lw=1.5, label="Choke")
    axes[2].set_ylabel("Choke (%)"); axes[2].set_xlabel("Time (hr)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Unmeasured WHP disturbance at t=60 hr -- controller recovers "
                 "on feedback alone", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RESULTS / "robustness_disturbance.png", dpi=130)
    plt.close(fig)

    # Worst-margin summary bar chart.
    labels, margins = [], []
    for section in ("noise_sweep", "model_mismatch", "disturbance"):
        for k, v in results[section].items():
            labels.append(k)
            margins.append(v["worst_margin_psi"])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["tab:green" if m > 0 else "tab:red" for m in margins]
    ax.barh(range(len(labels)), margins, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.axvline(0, color="red", lw=1.6)
    ax.set_xlabel("Worst pressure margin over the whole run (psi)  --  "
                  "negative = a violation occurred")
    ax.set_title("Robustness: tightest envelope margin per stress case "
                 "(all runs, all pressures)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(RESULTS / "robustness_margins.png", dpi=130)
    plt.close(fig)

    total_viol = sum(
        v.get("violations", v.get("violations_max_over_seeds", 0))
        for section in results.values() for v in section.values()
    )
    results["summary"] = {
        "total_cases": sum(len(s) for s in
                           (results["noise_sweep"], results["model_mismatch"],
                            results["disturbance"])),
        "total_violations_all_cases": int(total_viol),
        "min_margin_psi_all_cases": round(min(margins), 2),
    }
    (RESULTS / "robustness_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSummary: {results['summary']['total_cases']} stress cases, "
          f"{total_viol} total violations, "
          f"tightest margin {results['summary']['min_margin_psi_all_cases']} psi")
    print(f"Wrote {RESULTS / 'robustness_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
