# Sunburst Dashboard (Python + Plotly)

Generates an interactive HTML dashboard with a 6-level sunburst of user/navigation paths, a table of the path breakdown, and a horizontal bar chart of path counts. No server required — the result is a single HTML file.

**Tech:** Python 3.10+, Pandas, Plotly

## Screenshot
A preview of the generated dashboard (synthetic data):

After running the steps below, the full interactive HTML will be created at outputs/sunburst.html and can be opened in your browser.

## Quickstart

# 1) Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# 2) Install dependencies
pip install -r requirements.txt

# 3) Build the dashboard -> outputs/sunburst.html
python scripts/viz.py

# 4) Open in your browser (macOS)
open outputs/sunburst.html

Notes:
All numbers are synthetic and for demonstration only.

License:
MIT
