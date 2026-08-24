"""M11: AI-controlled electrochromic/SPD window tinting.

Why this exists: the project's first-round judges flagged that auto-dimming
SPD glass (reacting to its own built-in light sensor) isn't new -- it's an
existing product. The differentiator the team settled on is anticipatory
control: use the SAME "operational analysis" lookahead this project already
built for HVAC (M4's forecast features) to pre-empt sun exposure and tunnels
before they happen, not react to them after the fact. See ROADMAP.md's M11
section for the full brief and scope decisions.

DELIBERATELY RULE-BASED, NOT A TRAINED MODEL. Explicit scope guidance from
the team: "no one is going to look at the code... we just need a simulation
that shows our idea" (ROADMAP.md's M11 section). A trained model here would
cost real time for zero demo benefit -- the anticipatory-vs-reactive
CONTRAST is what's being demonstrated, and a two-line lookahead rule shows
it exactly as clearly as a neural net would, for a fraction of the effort.

Two controllers, mirroring ThermostatController/AnticipatorySetpointAdvisor's
reactive-baseline-vs-anticipatory-advisor shape on purpose -- same
demonstrable contrast, same reason it's convincing there: both react to sun,
only one of them reacts to sun that hasn't arrived yet.

Both take the EXISTING `ControllerInputs` (src/controllers.py) rather than a
new parallel struct -- they only read `ghi_w_m2`/`ghi_fcst_h` from it, but
`run_controller()` already builds one full ControllerInputs per minute for
the HVAC advisor, so reusing it means no second per-minute object and no new
type whose only job is to carry two floats.

TUNNELS FALL OUT OF THIS FOR FREE, NOT AS A SEPARATE CASE. There is no
explicit "in_tunnel" field anywhere in this file. A tunnel (or any other
route feature that blocks the sun) shows up simply as ghi_w_m2 / ghi_fcst_h
reading near zero for those minutes (see src/evaluate.py's tunnel-masking
note) -- the anticipatory advisor already "sees" an oncoming tunnel the
moment it enters the forecast horizon, the same way it already sees an
oncoming sunny stretch, with no special-case code for tunnels specifically.
"""

GHI_FULL_TINT_W_M2 = 900.0
"""[ASSUMPTION] -- GHI level at which the controller darkens fully.
Anchored against real fetched 2024 summer GHI maxima (Cairo 993, Aswan
1016 W/m^2 -- docs/PARAMETERS.md), a round number comfortably inside the
real observed range rather than its extreme tail. Not sourced from any
smart-glass product datasheet -- same honesty standard as every other
[ASSUMPTION] in this project."""


def _ghi_to_tint(ghi_w_m2: float) -> float:
    """Simple, explainable, linear -- deliberately not a trained model (see
    module docstring). 0.0 (clear) at no sun, 1.0 (fully darkened) at or
    above GHI_FULL_TINT_W_M2, linear between."""
    return max(0.0, min(1.0, ghi_w_m2 / GHI_FULL_TINT_W_M2))


class ReactiveTintController:
    """Baseline: mimics an off-the-shelf SPD panel's own built-in light
    sensor -- reacts to CURRENT sun only, so it cannot start darkening
    before direct sun hits or clear before a tunnel; it only ever responds
    once the change has already arrived. This is exactly the "SPD does it
    automatically" behaviour the judges said isn't new -- kept here so
    there is an honest reactive baseline to demonstrate the anticipatory
    advisor's actual contribution against, the same role ThermostatController
    plays for the HVAC setpoint.
    """

    def recommend_tint(self, inputs) -> float:
        return _ghi_to_tint(inputs.ghi_w_m2)


class AnticipatoryTintAdvisor:
    """Reacts to the FORECAST GHI at t+H (`ghi_fcst_h`, the same lookahead
    feature M4's forecaster and AnticipatorySetpointAdvisor already
    consume), not current sun -- so it can pre-darken ahead of a sunny
    stretch, or clear ahead of a tunnel, instead of only after the fact.
    Deliberately rule-based, not a trained model -- see module docstring.
    """

    def recommend_tint(self, inputs) -> float:
        return _ghi_to_tint(inputs.ghi_fcst_h)
