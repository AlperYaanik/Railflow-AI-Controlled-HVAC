"""Produces M6's required deliverable (ROADMAP.md: "a single sweep plot
exists -- saving vs tau_act") from src/sweep_tau_act.py's saved CSV.

Two panels, not one, because the sweep's most useful finding turned out not
to be the energy-saving curve alone: ThermostatController and the
proportional-only ablation both swing wildly in comfort across the sweep
(non-monotonically -- a lagged actuator can damp bang-bang's overshoot at
first, then just become too slow to track a disturbance), while
AnticipatorySetpointAdvisor stays comparatively stable. That robustness-to-
the-unknown-parameter claim needs the comfort panel to be visible at all;
the energy panel alone would hide it.
"""

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import DATA_DIR

CSV_PATH = DATA_DIR / "m6_tau_act_sweep.csv"
PLOT_PATH = DATA_DIR / "m6_tau_act_sweep.png"


def plot(df: pd.DataFrame | None = None, out_path=PLOT_PATH):
    df = df if df is not None else pd.read_csv(CSV_PATH)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(df["tau_act_min"], df["onoff_mean_saving_pct"], "o-", color="#1f77b4", label="mean")
    ax1.plot(df["tau_act_min"], df["onoff_median_saving_pct"], "s--", color="#1f77b4",
              alpha=0.5, label="median")
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_xlabel("tau_act (actuator lag, min) -- UNKNOWN, swept not asserted")
    ax1.set_ylabel("energy saving vs ThermostatController (%)")
    ax1.set_title("M5 headline: anticipatory vs on/off baseline")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(df["tau_act_min"], df["onoff_total_baseline_dh"], "o-",
              label="ThermostatController (on/off)", color="#d62728")
    ax2.plot(df["tau_act_min"], df["ff_total_baseline_dh"], "^-",
              label="Proportional-only (no forecast)", color="#ff7f0e")
    ax2.plot(df["tau_act_min"], df["onoff_total_candidate_dh"], "D-",
              label="AnticipatorySetpointAdvisor (shipped)", color="#2ca02c")
    ax2.set_xlabel("tau_act (actuator lag, min) -- UNKNOWN, swept not asserted")
    ax2.set_ylabel("total degree-hours outside band, K*h\n(23 test-split scenarios, lower = better)")
    ax2.set_title("Comfort robustness to the unknown lag")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("M6 -- sensitivity to actuator lag (tau_act), never claimed as fact for any real vehicle",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = plot()
    print(f"saved {path}")
