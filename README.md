# Autonomous Production Choke Controller — Honeywell Hackathon (Problem 3A)

An autonomous controller for a single naturally flowing oil well. Every 1-hour
control interval it sets the production choke (0–100 %) to **track a target oil
rate** while **never** violating the pressure operating envelope (lower bounds on
WHP / FLP / BHP) or the choke ramp limit (±5 %/step). When a target is
physically unsafe, it **settles at the maximum achievable safe rate** instead of
chasing it.

The controller is **safe-by-construction**: it predicts the consequence of every
candidate move with an identified dynamic model and rejects any move that would
breach the envelope. It knows when to say no.

## Results at a glance

| Scenario | Target | Settled rate | Envelope violations | Naive-baseline violations |
|---|---|---|---|---|
| A — Startup → Target | 130 bbl/hr | 129.9 bbl/hr | **0** | 0 |
| B — Target Tracking | 100 → 150 bbl/hr | 149.7 bbl/hr | **0** | 0 |
| C — Infeasible Target | 185 bbl/hr | 154.9 bbl/hr (max safe) | **0** | 168 |

A and B track their (feasible) targets to within 0.4 bbl/hr. C correctly refuses
the infeasible 185 bbl/hr request and lines out at the maximum safe rate with
zero violations, where an envelope-unaware baseline racks up 168 pressure
breaches. Numbers are reproduced by `python main.py` and written to
`results/metrics.json`.

## How to run

```bash
pip install -r requirements.txt
python main.py
```

This runs the full pipeline: loads the step-test data, identifies the models
(printing gains and time constants), runs the three scenarios closed-loop, writes
all plots and `metrics.json` to `results/`, and **asserts zero envelope
violations** across A/B/C (the run fails loudly if that ever regresses).

The narrative walk-through is in
[`notebook/choke_control_solution.ipynb`](notebook/choke_control_solution.ipynb),
and the technical write-up is in [`report.md`](report.md).

## Approach

1. **Dynamic model identification (`model_id.py`).** The provided 120-hour
   open-loop step test is fit with a first-order **ARX(1)** model per output via
   ordinary least squares:

   ```
   y[k+1] = a·y[k] + (1 − a)·(b0 + b1·u[k])
   ```

   which is linear in the parameters. This yields the steady-state gain
   `b1 = dy_ss/du`, intercept `b0`, and time constant `τ = −Ts/ln(a)` for each of
   Q, WHP, FLP, BHP. A free-run (multi-step) simulation validates the fit to
   1.5–2.4 % of signal span. Residual scatter sets the simulator's noise level.

2. **Operating envelope (`envelope.py`).** All constraints live in one place:
   `WHP_min = 215`, `FLP_min = 150`, `BHP_min = 2850` psi, choke ∈ [0, 100] %,
   |Δchoke| ≤ 5 %/step. Opening the choke raises rate but lowers every pressure,
   so the binding constraints are lower bounds — WHP binds first, at choke
   ≈ 65.8 % (hard max safe rate ≈ 159 bbl/hr). These numbers are a documented
   assumption (the brief gives none) and are trivially editable when the official
   envelope is announced.

3. **Predictive controller (`controller.py`).** A simplified MPC by brute-force
   candidate search:
   - enumerate reachable next positions (choke ± up to 5 %, fine grid);
   - predict each candidate's pressure trajectory over a settling horizon **and**
     its steady state with the ARX models;
   - **reject** any candidate that would push a pressure below its bound plus a
     noise-sized safety margin;
   - among safe candidates, minimise `(Q_ss − target)² + λ·(Δchoke)²`.

   Because a first-order response moves monotonically toward its steady state,
   proving steady-state feasibility (plus margin) proves the whole trajectory
   safe. The same cost that tracks a reachable target automatically settles at
   the highest safe rate when the target is unreachable, and the controller backs
   the choke off if the plant is ever already outside the envelope.

## Swapping in the official simulator

`simulator.WellSimulator` is a stand-in with the exact interface the brief
specifies:

```python
Q, WHP, FLP, BHP = simulator.step(choke_position)
```

To use the official event simulator instead, replace the `WellSimulator(...)`
construction in `scenarios.run_closed_loop` with the official object — the
controller and scenario loop are unchanged. If the official envelope differs,
edit only `envelope.py`.

## File layout

```
.
├── dataio.py        # load & tidy the step-test dataset
├── model_id.py      # ARX(1) identification + free-run validation
├── envelope.py      # operating envelope + constraints (single source of truth)
├── simulator.py     # WellSimulator plant stand-in (official .step() signature)
├── controller.py    # ChokeMPC safety-first predictive controller + naive baseline
├── plotting.py      # 6-trend scenario plots + model-validation plot
├── scenarios.py     # Scenario A/B/C runners + metrics
├── main.py          # runs everything; asserts 0 violations; writes results/
├── data/            # provided step-test CSV
├── results/         # generated PNGs + metrics.json
├── notebook/        # narrative Jupyter walk-through
├── submission/      # deck PDF, report PDF, and the <10 MB ZIP bundle
├── IDEA_Presentation_Format.pptx   # filled presentation deck
├── report.md        # technical report
└── requirements.txt
```
