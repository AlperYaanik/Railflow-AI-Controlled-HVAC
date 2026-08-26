"""M14: the power-vs-time comparison figure.

Plots one journey under one scenario, baseline vs AI, with the disturbance
window shaded so a viewer can see WHEN the environment changed rather than
inferring it from a curve's shape. Both arms fly through the identical
environment -- the scenario is passed to both run_controller() calls.

Nothing here post-processes a result. The curves are whatever the
controllers produced; if the AI's power trace is not smoother, the chart
shows that (and on this project's measurements it frequently is not -- see
ROADMAP.md's M14 section on the ramp metric).
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas (see src/train.py)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import scenarios as scenarios_mod
from src.compare_controllers import _load_booster, held_out_test_scenarios
from src.config import DATA_DIR, load_config
from src.controllers import (
    SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN,
    AnticipatorySetpointAdvisor,
    ThermostatController,
)
from src.evaluate import run_controller, score
from src.features import LiveFeatureBuilder


CLEAREST_COMBINED_JOURNEY = 1
"""TEST journey where the forecast's effect under `combined` is largest
(comfort 1.42 -> 1.26 K*h, an 11.6% improvement on this journey against the
+3.12% aggregate across all 23). Used as the DEFAULT for the figure because
a chart has to show a mechanism to be worth showing, and on a median journey
the two traces overlap almost exactly. This is disclosure, not selection:
the aggregate over every journey -- including the ones where the AI does
nothing or does worse -- is what ROADMAP.md's M14 result table reports, and
`build_figure(scenario_id=...)` will plot any of them."""


def build_figure(scenario_name="combined", ff_weight=0.3,
                 scenario_id=CLEAREST_COMBINED_JOURNEY, cfg=None):
    cfg = cfg if cfg is not None else load_config()
    booster = _load_booster()
    scenario = scenarios_mod.get(scenario_name)
    rows = held_out_test_scenarios()
    row = rows[rows["scenario_id"] == scenario_id].iloc[0]

    run_kwargs = dict(cfg=cfg, city=row.city, date=str(row.date), depart_hour=row.depart_hour,
                      direction=row.direction, pattern=row.pattern, load_factor=row.load_factor,
                      advisor_update_interval_min=SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN,
                      scenario=scenario)

    def ai_factory(model):
        builder = LiveFeatureBuilder(cfg, city=row.city, direction=row.direction,
                                     pattern=row.pattern, load_factor=row.load_factor,
                                     depart_hour=row.depart_hour)
        return AnticipatorySetpointAdvisor(model, booster, builder, ff_weight=ff_weight)

    base = run_controller(lambda m: ThermostatController(m), **run_kwargs)
    ai = run_controller(ai_factory, **run_kwargs)
    bs, as_ = score(base, cfg), score(ai, cfg)

    fig, (ax_t, ax_p, ax_e) = plt.subplots(
        3, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1.4]})

    for ax in (ax_t, ax_p, ax_e):
        for start, end, label in scenario.disturbance_windows():
            ax.axvspan(start, min(end, len(base) - 1), color="#d62728", alpha=0.10,
                       label=label if ax is ax_t else None)

    band_k = cfg["comfort"]["band_k"]
    ax_t.fill_between(base["t"], base["setpoint_c"] - band_k, base["setpoint_c"] + band_k,
                      color="green", alpha=0.10, label=f"comfort band (±{band_k:g} K)")
    ax_t.plot(base["t"], base["setpoint_c"], "--", color="gray", linewidth=1, label="setpoint")
    ax_t.plot(base["t"], base["t_air_c"], color="#d62728", linewidth=1.5, label="static schedule")
    ax_t.plot(ai["t"], ai["t_air_c"], color="#2ca02c", linewidth=1.5,
              label=f"AI (forecast on, ff_weight={ff_weight})")
    ax_t.set_ylabel("cabin air temp (°C)")
    ax_t.set_title(
        f"Scenario '{scenario_name}' — journey #{scenario_id} ({row.city}, {row.date}), "
        f"the clearest of the 23\nboth controllers fly the identical disturbed environment; "
        f"comfort {bs.degree_hours:.2f} → {as_.degree_hours:.2f} K·h "
        f"({100 * (1 - as_.degree_hours / bs.degree_hours):+.1f}%), "
        f"energy {100 * (1 - as_.energy_kwh / bs.energy_kwh):+.1f}%",
        fontsize=11)
    ax_t.legend(loc="upper right", fontsize=8)
    ax_t.grid(alpha=0.3)

    ax_p.plot(base["t"], base["electrical_w"] / 1000.0, color="#d62728", linewidth=1.3,
              label=f"static schedule (peak {bs.peak_power_kw:.1f} kW, "
                    f"ramp {bs.power_ramp_w_per_min:.0f} W/min)")
    ax_p.plot(ai["t"], ai["electrical_w"] / 1000.0, color="#2ca02c", linewidth=1.3,
              label=f"AI (peak {as_.peak_power_kw:.1f} kW, ramp {as_.power_ramp_w_per_min:.0f} W/min)")
    ax_p.set_ylabel("electrical draw (kW)")
    ax_p.legend(loc="upper left", fontsize=8)
    ax_p.grid(alpha=0.3)

    ax_e.plot(base["t"], (base["electrical_w"].cumsum() / 60000.0), color="#d62728",
              linewidth=1.3, label=f"static schedule ({bs.energy_kwh:.2f} kWh)")
    ax_e.plot(ai["t"], (ai["electrical_w"].cumsum() / 60000.0), color="#2ca02c",
              linewidth=1.3, label=f"AI ({as_.energy_kwh:.2f} kWh)")
    ax_e.set_ylabel("cumulative\nenergy (kWh)")
    ax_e.set_xlabel("minute")
    ax_e.legend(loc="upper left", fontsize=8)
    ax_e.grid(alpha=0.3)

    fig.tight_layout()
    return fig, bs, as_


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "combined"
    fig, bs, as_ = build_figure(scenario_name=name)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"m14_{name}.png"
    fig.savefig(out, dpi=140)
    print(f"saved {out}")
    print(f"  static: energy={bs.energy_kwh:.2f} kWh  degree_hours={bs.degree_hours:.2f}  "
          f"peak={bs.peak_power_kw:.1f} kW  ramp={bs.power_ramp_w_per_min:.0f} W/min")
    print(f"  AI    : energy={as_.energy_kwh:.2f} kWh  degree_hours={as_.degree_hours:.2f}  "
          f"peak={as_.peak_power_kw:.1f} kW  ramp={as_.power_ramp_w_per_min:.0f} W/min")
