# Premier League Match Outcome Predictor (H/D/A)

Predicts **Home / Draw / Away** results using scikit-learn on FBRef team logs with **rolling form** features and a **time-aware** split.

## Data
Use an FBRef team log CSV with columns like:
`date, time, comp, round, day, venue, result, gf, ga, opponent, sh, sot, xg, ...`
Place it as `data/matches.csv` (this folder is gitignored). The script converts **team logs → match-level** internally.

## Setup

### Create a virtual environment (recommended)
**Windows (PowerShell / VS Code Terminal)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate

Minimal dependencies required

python -m pip install -U pip
python -m pip install pandas scikit-learn numpy
