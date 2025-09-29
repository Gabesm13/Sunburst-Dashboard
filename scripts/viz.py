#!/usr/bin/env python3
"""
viz.py — Build an interactive HTML with a 6-level sunburst in the (1,1) cell.
Build a barplot of path data in the (1,2) and (1,3) cells.
"""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------
COLOR_MAP = {
    "Page 1": "#003f5c",
    "Page 2": "#bc5090",
    "Page 3": "#4cc3ff",
}

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def load_data(root: Path):
    """
    Load and preprocess sunburst and bar data from CSV files.
    """
    data_dir = root / "data"

    # Load sunburst hierarchy
    sb_df = pd.read_csv(data_dir / "sunburst_full_hierarchy.csv")
    sb_df["parent"] = sb_df["parent"].fillna("")

    # Derive root and colors
    sb_df["root"] = sb_df["id"].str.split(" - ").str[0]
    sb_df["color"] = sb_df["root"].map(COLOR_MAP)

    # Extract page number for sorting
    sb_df["page_num"] = (
        sb_df["label"]
        .str.extract(r"Page\s+(\d+)", expand=False)
        .astype(int)
    )

    # Sort hierarchy for consistent ordering
    sb_df = (
        sb_df
        .sort_values(["parent", "page_num"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # Load bar-level data
    bar_df = pd.read_csv(data_dir / "path_bar_data.csv")
    bar_df = (
        bar_df
        .sort_values(by=["level1", "level2", "level3", "level4", "level5", "level6"])
        .reset_index(drop=True)
    )
    bar_df["color"] = bar_df["level1"].map(COLOR_MAP)

    return sb_df, bar_df


def compute_bar_records(sb_df: pd.DataFrame):
    """
    Compute leaf and remainder records for the bar chart.
    """
    records = []

    # Identify internal nodes
    internal_mask = sb_df.apply(lambda r: (sb_df.parent_id == r["id"]).any(), axis=1)

    # a) Real leaves
    leaves = sb_df.loc[~internal_mask]
    for _, leaf in leaves.iterrows():
        parts = leaf["id"].split(" - ")
        parts += [""] * (6 - len(parts))
        rec = {f"level{i+1}": parts[i] for i in range(6)}
        rec["count"] = leaf["value"]
        records.append(rec)

    # b) Remainders for each internal node
    for _, node in sb_df.loc[internal_mask].iterrows():
        child_sum = sb_df.loc[sb_df.parent_id == node["id"], "value"].sum()
        rem = node["value"] - child_sum
        if rem > 0:
            parts = node["id"].split(" - ")
            parts += [""] * (6 - len(parts))
            rec = {f"level{i+1}": parts[i] for i in range(6)}
            rec["count"] = rem
            records.append(rec)

    bar_records = pd.DataFrame(records)
    bar_records = (
        bar_records
        .sort_values(by=["level1","level2","level3","level4","level5","level6"])
        .reset_index(drop=True)
    )

    # Build path and color
    bar_records["path"] = bar_records[
        ["level1","level2","level3","level4","level5","level6"]
    ].apply(lambda r: " → ".join([p for p in r if p]), axis=1)
    bar_records["root"] = bar_records["path"].str.split(" → ").str[0]
    bar_records["color"] = bar_records["root"].map(COLOR_MAP)

    return bar_records


def create_figure(sb_df: pd.DataFrame, bar_df: pd.DataFrame):
    """
    Build the 1x3 subplot figure with sunburst, table, and bar chart.
    """
    # Create subplots layout
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type":"domain"}, {"type":"table"}, {"type":"xy"}]],
        column_widths=[0.5, 0.25, 0.25],
        horizontal_spacing=0.01,
        subplot_titles=[
            "<u><b>6-Level User Navigation Sunburst</b></u>",
            "<b>User Navigation Breakdown</b>"
        ]
    )

    # Sunburst trace
    fig.add_trace(
        go.Sunburst(
            ids=sb_df["id"],
            parents=sb_df["parent_id"],
            labels=sb_df["label"],
            values=sb_df["value"],
            branchvalues="total",
            marker=dict(colors=sb_df["color"]),
            hovertemplate="%{label}: %{value}<extra></extra>",
            rotation=90,
            sort=False,
            texttemplate=(
                "<span style='font-size:30px; font-family:Arial;'>"
                "%{value}</span><br>"
                "<span style='font-size:12px; color:'white'>%{label}</span>"
            ),
            insidetextorientation="radial"
        ),
        row=1, col=1
    )

    # Table trace
    header_vals = ['Level 1','Level 2','Level 3','Level 4','Level 5','Level 6']
    cell_rows = [bar_df[col].fillna('').tolist() for col in [
        'level1','level2','level3','level4','level5','level6'
    ]]
    fig.add_trace(
        go.Table(
            header=dict(
                values=header_vals,
                fill_color='lightgrey',
                align='left',
                font=dict(size=18)
            ),
            cells=dict(
                values=cell_rows,
                align='left',
                font=dict(size=18)
            )
        ),
        row=1, col=2
    )

    # Bar chart trace
    fig.add_trace(
        go.Bar(
            x=bar_df['count'],
            y=bar_df['path'],
            orientation='h',
            marker=dict(color=bar_df['color']),
            hovertemplate="%{y}: %{x}<extra></extra>",
            showlegend=False
        ),
        row=1, col=3
    )
    # Axes updates
    fig.update_xaxes(
        title_text='Clicks', domain=[0.745, 1], row=1, col=3
    )
    fig.update_yaxes(
        automargin=True, autorange='reversed', domain=[0.05, 0.974], row=1, col=3
    )

    # Layout polish
    fig.update_layout(
        width=3500, height=1700,
        margin=dict(l=40, r=40, t=130, b=40),
    )

    # Adjust subplot titles
    fig.layout.annotations[0].update(
        font=dict(size=65), xref='paper', yref='paper', x=0.17, y=1.02
    )
    fig.layout.annotations[1].update(
        font=dict(size=36), xref='paper', yref='paper', x=0.63, y=1.01
    )

    # Adjust table row heights dynamically
    n_rows = len(bar_df)
    avail_px = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b
    rows_total = n_rows + 1  # +1 for header
    row_height = avail_px / rows_total * 0.95
    table_trace = next(trace for trace in fig.data if isinstance(trace, go.Table))
    table_trace.cells.height = row_height
    table_trace.header.height = row_height

    # Fake Legend via Annotations
    legend_x0 = 0.02
    legend_y0 = 0.93
    dx = 0.034

    for i, (root, color) in enumerate(COLOR_MAP.items()):
        fig.add_annotation(
            x=legend_x0 + i * dx,
            y=legend_y0,
            xref="paper", yref="paper",
            showarrow=False,
            align="left",
            text=(
                f"<span style='font-size:40px;"
                f"color:{color};'>&#9632;</span> "
                f"<span style='font-size:20px;color:black;'>{root}</span>"
            ),
            xanchor="left",
            yanchor="bottom"
        )

    fig.add_annotation(
        text=(f"<span style='font-size:20px;color:black;'>Beginning of Nav Path</span>"),
        xref="paper", yref="paper",
        x=legend_x0+.014,
        y=legend_y0+.05,
        showarrow=False,
    ),

    pad_x = 0.01
    pad_y = 0.01
    n = len(COLOR_MAP)

    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=legend_x0 - pad_x,
        y0=legend_y0 - pad_y,
        x1=legend_x0 + n * dx + pad_x,
        y1=legend_y0 + 0.05 + pad_y,
        fillcolor="white",
        line=dict(color="black", width=2),
        layer="below"
    )

    return fig


def main():
    # 1. Locate project folders
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # 2. Load and preprocess data
    sb_df, initial_bar_df = load_data(project_root)

    # 3. Compute bar records including leaves and remainders
    bar_df = compute_bar_records(sb_df)

    # 4. Create figure and export
    fig = create_figure(sb_df, bar_df)
    outpath = outputs_dir / "sunburst.html"
    fig.write_html(outpath)
    print(f"✓ Sunburst dashboard written to {outpath}")


if __name__ == "__main__":
    main()
