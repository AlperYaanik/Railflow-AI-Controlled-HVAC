# M7 fallback: recording script for the live demo

**Why this exists.** ROADMAP.md's M7 milestone requires a screen recording of
`src/app.py` as a fallback in case the live Streamlit demo fails in front of
the jury (network hiccup, projector/display issue, a stale `data/`/`models/`
directory on the presentation machine, etc.). This script is the
walkthrough to follow when recording it -- written so the recording takes
one take, ~90 seconds, and hits every point a live demo would.

**Updated after M9 added a third chart panel (cabin humidity).** The
original version of this script only described two panels (temperature,
HVAC command) -- found stale during a full-system verification pass, fixed
before anyone tried to record against it.

**Updated again after M10 reframed the middle panel.** Railflow's compute
cannot touch a real HVAC unit's control electronics (would void the
manufacturer's warranty), so the system no longer commands power -- it
recommends a cabin setpoint. The middle panel now plots that recommendation
(°C), not a watts command; the narration line below was rewritten to match.

**Why a script and not an already-recorded file.** No tool available in this
environment can capture the presenter's own screen: the Chrome extension
that could drive a real browser recording wasn't connected here, this
project's sandboxed browser preview pane isn't compositing frames
(`screenshot failed: the Browser pane is not displayed`), and no local
`ffmpeg` is installed to assemble one from still frames. All three were
tested directly, not assumed. A jury-facing recording benefits from a human
narrating it anyway -- a silent, automated walkthrough wouldn't explain
what's on screen the way a presenter's own voice does.

## Recording tool

**Xbox Game Bar** (`Win`+`G`) -- built into Windows 11, no install needed.
Records screen + microphone to an `.mp4` under `Videos\Captures`. Start it,
click the record button (or `Win`+`Alt`+`R`), narrate the script below, stop
with the same shortcut. If a more polished result is wanted later (trimming,
captions), OBS Studio is the standard free option, but Game Bar is enough
for a fallback.

## Before recording

1. Run the full setup once so nothing stalls mid-recording:
   ```bash
   python -m src.weather && python -m src.data_generator && python -m src.train
   ```
2. Launch the demo and confirm it's already showing data before you hit
   record -- `streamlit run src/app.py`, or the `railflow-demo` config in
   `.claude/launch.json`.
3. Maximize the browser window; close other tabs/notifications that could
   pop up during the take.

## The walkthrough (~90s)

**0:00-0:10 -- Title and framing.**
Let the page finish loading. Read the caption aloud or paraphrase it:
> "This is the same simulator and the same two controllers the project's
> results were built and tested on -- not a separate demo path. If the
> numbers here ever disagreed with the written comparison, that would be a
> bug in the app, not a new result."

**0:10-0:30 -- The scenario picker.**
Point at the sidebar. Open **Pick a journey** and select a different
scenario than the default.
> "All 23 scenarios here are from the held-out test split -- eleven days of
> real Cairo and Aswan weather the forecaster never trained or validated on.
> Switching scenarios re-runs the full simulation for both controllers."

**0:30-0:50 -- The four metrics.**
Point at each metric column in turn.
> "Energy, on the left, is the on/off thermostat baseline -- a realistic
> baseline with a sourced hysteresis band, not a strawman. Next to it, the
> anticipatory controller's energy, with the percentage saved. Comfort is
> degree-hours outside the setpoint band -- lower is better -- baseline
> first, then anticipatory."

Do **not** read out a specific saving percentage as if it's the headline
number for every scenario -- it varies scenario to scenario by design (see
ROADMAP.md's M5 section for the aggregate, honest number). Say what the
metric *is*, let the on-screen number for whichever scenario is showing
speak for itself.

**0:50-1:05 -- Playback.**
Drag the **Minute** slider partway, then toggle **Auto-play from here** and
let it run a few seconds.
> "The chart isn't a static image -- it's the full minute-by-minute
> trajectory. Red is on/off, green is anticipatory, the shaded band in the
> top panel is the comfort tolerance. The middle panel is the cabin
> setpoint each controller recommended -- not a power command. A separate,
> unmodified on/off unit, simulated identically for both controllers, is
> what actually turns that recommendation into on/off cycling."

**1:05-1:25 -- The humidity panel.**
Point at the bottom panel while it's still auto-playing or after it stops.
> "This bottom panel tracks cabin humidity -- something most HVAC demos
> don't show at all. It's an honest upper bound, not a fully corrected
> number: the simulation doesn't yet model the AC actively removing
> moisture while it cools, and that's stated on screen, not hidden in a
> footnote."
Do not linger on this beat -- one sentence is enough. It exists to show the
project discloses its own known limitations, not to explain the physics.

**1:25-1:35 -- What went into it.**
Expand **What this scenario looked like going in**.
> "And this is the exact journey behind those numbers -- city, date,
> direction, stopping pattern, load factor -- so nothing here is
> hand-picked or hidden."

## Honesty reminder for the narration

Same rule as everywhere else in this project (ROADMAP.md's Honesty rules):
never say "X% savings on a real train" while narrating. The correct framing
is "X% within the same comfort band on a parameterised cabin model" --
state plainly, if asked, that the data is synthetic and no real rail data
was available.

## After recording

Save the `.mp4` somewhere the presentation machine has offline (a USB
stick or the laptop's local disk, not only cloud storage) -- the entire
point is that it doesn't depend on anything the live demo depends on. It is
**not** committed to this repo (git is for code and regenerable artifacts,
not a multi-minute video file); keep it alongside the slide deck instead.
