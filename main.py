"""End-to-end pipeline for the Autonomous Production Choke Controller (Problem 3A).

Running ``python main.py`` will:

  1. Load and validate the step-test dataset.
  2. Identify the ARX(1) dynamic models (prints gains + time constants) and save
     a model-validation plot.
  3. Run the three demonstration scenarios (A, B, C) closed-loop with the
     safety-first MPC, saving the required 6-trend plot for each.
  4. Run the naive baseline for comparison and save a baseline-vs-MPC plot.
  5. Write results/metrics.json.
  6. ASSERT zero envelope violations across A, B and C (fails loudly otherwise).

All artefacts land in ``results/`` and are kept small for the 10 MB submission
limit.
"""
from __future__ import annotations

import json
from pathlib import Path

from dataio import infer_sample_time, load_step_test
from envelope import DEFAULT_ENVELOPE
from model_id import format_report, identify_all, validation_fit
from plotting import plot_baseline_comparison, plot_model_validation, plot_scenario
from scenarios import run_all_baseline, run_all_mpc

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    env = DEFAULT_ENVELOPE

    # 1. Data ------------------------------------------------------------
    print("=" * 70)
    print("Autonomous Production Choke Controller -- Problem 3A")
    print("=" * 70)
    df = load_step_test()
    ts = infer_sample_time(df)
    print(f"Loaded step-test: {len(df)} rows, Ts = {ts} hr\n")

    # 2. Model identification -------------------------------------------
    models = identify_all(df, Ts=ts)
    print(format_report(models))
    print("\nFree-run model validation (open-loop vs data):")
    model_val = {}
    for name, m in models.items():
        v = validation_fit(m, df)
        model_val[name] = {k: round(val, 3) for k, val in v.items()}
        print(f"  {name:<4} RMSE={v['rmse']:.2f} ({v['rmse_pct_of_span']:.1f}% of span)")
    val_plot = plot_model_validation(models, df, RESULTS / "model_validation.png")
    print(f"  -> {val_plot.relative_to(HERE)}")

    # Report the envelope consequence for context.
    whp = models["WHP"]
    q = models["Q"]
    u_whp_bind = (whp.b0 - env.WHP_min) / (-whp.b1)
    max_safe_rate = q.steady_state(u_whp_bind)
    print(
        f"\nEnvelope: WHP hits {env.WHP_min} psi at choke {u_whp_bind:.1f}% "
        f"-> hard max safe rate ~{max_safe_rate:.0f} bbl/hr (WHP binds first)."
    )

    # 3. Closed-loop scenarios (safety-first MPC) -----------------------
    print("\n" + "-" * 70)
    print("Closed-loop scenarios (safety-first MPC):")
    mpc = run_all_mpc(models, envelope=env)
    for name in ("A", "B", "C"):
        res = mpc[name]
        p = plot_scenario(res, env, RESULTS / f"scenario_{name}.png")
        m = res.metrics
        print(
            f"  {name}: violations={m['violations_total']}, "
            f"settled Q={m['settled_oil_rate_bbl_hr']} "
            f"(target {m['final_target_bbl_hr']:.0f}), "
            f"worst margin={m['worst_pressure_margin_psi']} psi  -> {p.name}"
        )

    # 4. Naive baseline for comparison ----------------------------------
    print("\nNaive baseline (envelope-unaware) for comparison:")
    naive = run_all_baseline(models, envelope=env)
    for name in ("A", "B", "C"):
        print(f"  {name}: violations={naive[name].metrics['violations_total']}")
    cmp_plot = plot_baseline_comparison(
        mpc["C"], naive["C"], env, RESULTS / "baseline_comparison_C.png"
    )
    print(f"  -> {cmp_plot.name}")

    # 5. Metrics ---------------------------------------------------------
    metrics = {
        "problem": "3A -- Autonomous Production Choke Controller",
        "sample_time_hr": ts,
        "envelope": {
            "WHP_min": env.WHP_min, "FLP_min": env.FLP_min, "BHP_min": env.BHP_min,
            "choke_min": env.choke_min, "choke_max": env.choke_max,
            "max_ramp_pct_per_step": env.max_ramp,
        },
        "models": {
            name: {
                "a": round(m.a, 4), "gain_b1": round(m.b1, 4), "b0": round(m.b0, 3),
                "tau_hr": round(m.tau_hr, 2), "noise_std": round(m.resid_std, 3),
            }
            for name, m in models.items()
        },
        "model_validation": model_val,
        "hard_max_safe_rate_bbl_hr": round(float(max_safe_rate), 1),
        "scenarios_mpc": {name: mpc[name].metrics for name in ("A", "B", "C")},
        "scenarios_baseline": {
            name: {
                "violations_total": naive[name].metrics["violations_total"],
                "violations_by_pressure": naive[name].metrics["violations_by_pressure"],
                "settled_oil_rate_bbl_hr": naive[name].metrics["settled_oil_rate_bbl_hr"],
            }
            for name in ("A", "B", "C")
        },
    }
    metrics_path = RESULTS / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {metrics_path.relative_to(HERE)}")

    # 6. Hard verification ----------------------------------------------
    total_violations = sum(mpc[n].metrics["violations_total"] for n in ("A", "B", "C"))
    baseline_violations = sum(naive[n].metrics["violations_total"] for n in ("A", "B", "C"))
    print("\n" + "=" * 70)
    print(f"MPC total envelope violations across A/B/C : {total_violations}")
    print(f"Naive baseline total violations across A/B/C: {baseline_violations}")
    print("=" * 70)

    assert total_violations == 0, (
        f"SAFETY FAILURE: MPC produced {total_violations} envelope violations "
        "(expected 0)."
    )
    # A and B targets are feasible -> require good tracking.
    for name in ("A", "B"):
        err = abs(mpc[name].metrics["steady_state_error_bbl_hr"])
        assert err <= 2.0, f"Scenario {name} steady-state error {err} bbl/hr exceeds 2.0."
    print("VERIFIED: 0 violations across all scenarios; A/B track target within 2 bbl/hr.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
