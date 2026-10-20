from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

COLORS = {"round-robin": "#9aa0a6", "least-loaded": "#4285f4", "cache-aware": "#e60023"}
MODELED = "MODELED latency (simulator), not measured on GPUs"

plt.rcParams.update({
    "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})
