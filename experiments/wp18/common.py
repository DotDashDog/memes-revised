"""WP-18 shared constants."""
from pathlib import Path

REPO = Path("/home/linus/memes-revised")
DATA = REPO / "dataset_saves/reddit/submissions-2025-12"
OUT = REPO / "experiments/WP-18"
SEED = 18016

BACKBONE = ["memes", "dankmemes", "meirl", "comedyheaven"]
EXTENSION = ["whenthe", "196", "shitposting"]
SUBS = BACKBONE + EXTENSION

# categorical palette (dataviz skill default, fixed order; colour follows the community)
COLORS = dict(zip(SUBS, ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2"]))


def parquet_path(sub: str) -> Path:
    return DATA / ("submissions.parquet" if sub == "memes" else f"{sub}_submissions.parquet")
