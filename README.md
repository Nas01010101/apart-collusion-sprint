# Apart AI Collusion Research Sprint, October 2026

Code, results and papers for two entries by Anas Elghoudane (Université de Montréal; Zarqa Lab, UdeM AI Club).

## Rosenhan (Track 3.2): overseers judging an undercover auditor

Paper: `report/rosenhan.md`, PDF in `report/build/`.

- `src/exp_rosenhan.py` plays the Liar's Bar games; `src/rate_rosenhan.py` runs the overseer conditions.
- `src/stats_rosenhan.py`, `src/expo_rosenhan.py` and `src/rosenhan_reading.py` compute the reported statistics.
- `src/rosenhan_figures.py` draws the figures into `figures/`.
- Game logs and ratings are in `results/rosenhan/`. The preregistration is `notes/rosenhan_plan.md`.

## Dolos-1.5B (Track 2): auditing a fine-tune with an undisclosed behaviour

Paper: `report/dolos.md`, PDF in `report/build/`.

- The scripts in `redteam/` run the flip tests, the WMDP, MMLU-Pro and MASK tests, the monitor test, weight diffs and weight swaps.
- `redteam/figs.py` draws the figures. Result files are in `redteam/results/`.

## Running

Python 3.12 with `uv`. GPU runs for the Rosenhan games used Modal. Tests: `pytest tests/`.
