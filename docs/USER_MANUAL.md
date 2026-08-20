# Railflow — user manual

A plain-English guide to installing this project and running the live demo.
For the technical details (why things are built this way) see `README.md`;
for project status and decisions see `ROADMAP.md`. This page is just: what
do I type, and what am I looking at.

## What this project is

Railflow predicts a train cabin's temperature a few minutes ahead and turns
the HVAC on/off earlier than a normal thermostat would. The demo shows two
controllers running side by side on the same simulated train journey: the
old way (a simple on/off thermostat) and the new way (the anticipatory
controller). You compare their energy use and comfort on the same trip.

All data is simulated — real weather, a physically modelled cabin, but no
real train was involved. That's a stated limitation, not a secret.

## 1. What you need

- Python 3.10 or newer installed
- A terminal (Command Prompt, PowerShell, or a terminal inside VS Code)
- Internet access, only for the one-time setup step below

## 2. Install (one time)

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

This installs everything the project needs — pandas, LightGBM (the
forecasting model library), Streamlit (the demo app), and a few others.

## 3. Build the data and the model (one time)

Nothing under `data/` or `models/` is stored in the project — you build it
locally, once, in this exact order:

```bash
python -m src.weather
python -m src.data_generator
python -m src.train
```

| Command | What it does | Roughly how long |
|---|---|---|
| `python -m src.weather` | Downloads real 2024 hourly weather for Cairo and Aswan | a few seconds |
| `python -m src.data_generator` | Simulates ~190 train journeys to build a training dataset | under a minute |
| `python -m src.train` | Trains the forecasting model on that dataset | a few seconds |

You only need to do this again if you delete the `data/` or `models/`
folders, or if someone changes the simulation code in a way that means the
old data/model no longer match it.

## 4. Run the live demo

```bash
streamlit run src/app.py
```

This opens a browser tab automatically (usually at `http://localhost:8501`).
If it doesn't open by itself, copy that address into your browser.

### What you'll see

**Sidebar — "Scenario"**
A dropdown called **Pick a journey**. There are 23 journeys to choose from.
Every one of them is a day the model never saw while it was being trained
— so the comparison you're looking at is always a fair, "unseen data" test,
not the model being tested on something it already memorised.

**Sidebar — "Playback"**
A **Minute** slider — drag it to any point in the journey to see the cabin's
state at that moment. Below it, an **Auto-play from here** checkbox — tick
it and the trip plays forward automatically from wherever the slider is.

**The four numbers at the top**
| Metric | Meaning |
|---|---|
| Energy — on/off | How much electricity the old-style thermostat used on this trip |
| Energy — anticipatory | How much the new controller used, and the % saved next to it |
| Comfort — on/off | How far outside the comfort temperature band the old controller drifted, in total (lower is better) |
| Comfort — anticipatory | Same, for the new controller |

**The chart**
Top panel: cabin temperature over time for both controllers (red = on/off,
green = anticipatory), with the comfort band shaded. Middle panel: what
each controller was actually commanding the HVAC to do, minute by minute.
Bottom panel: cabin humidity (%RH) for both controllers — a caption under
the chart explains why it isn't part of the AC's energy/comfort score (the
simulation doesn't yet model the AC removing moisture while it cools, so
this panel is a disclosed upper bound, not a fully corrected number).

**"What this scenario looked like going in" (click to expand)**
The exact journey details behind the numbers above — which city, which
date, which direction, how many passengers. Nothing on this page is
hand-picked; you can check any of the 23 journeys yourself.

### Closing the demo

Go back to the terminal and press `Ctrl+C`.

## 5. If something goes wrong

The app is built to explain itself rather than crash. If you see a red
error box at the top of the page instead of the dashboard, it will tell you
exactly which one-time setup step (Section 3) to (re-)run. A few examples:

| What the app says | What to do |
|---|---|
| "No trained model at ..." | Run `python -m src.train` |
| "Missing data file: ...scenarios_raw.parquet ..." | Run `python -m src.data_generator` (and `python -m src.weather` first if that hasn't run either) |
| "Missing data file: ...weather_..._summer.csv ..." | Run `python -m src.weather` |

If you see something other than a clean red message — a wall of technical
text (a "traceback") — that's a real bug, not a setup problem. Take a
screenshot and send it along.

## 6. Other things you can run (optional, for the curious)

You don't need any of these to see the demo — they're the underlying
scripts the demo itself calls, useful if you want the same comparison as
plain numbers/text instead of an interactive page:

```bash
python -m src.compare_controllers   # prints the headline energy/comfort comparison
python -m src.sweep_tau_act         # re-runs it across a range of assumed HVAC response delays
python -m pytest tests/ -q          # runs the full automated test suite (229 checks)
```

## 7. Where to look for more

- `README.md` — the technical setup/run reference, written for developers
- `ROADMAP.md` — what's been built, what was found along the way, and what's still open
- `docs/PARAMETERS.md` — every physical assumption behind the simulation, and where it came from
- `docs/demo_recording_script.md` — the backup plan: a script for recording a video of this demo, in case the live version can't be shown on presentation day
