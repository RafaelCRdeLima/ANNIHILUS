"""Plot style for the paper figures."""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2b8b7", "#e34948", "#a32d2c"])
REDS = LinearSegmentedColormap.from_list(
    "seq_red", ["#fbe9e8", "#f5c4c3", "#ee9796", "#e86a69", "#e34948", "#c13a39", "#8e2827"])
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def style():
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "lines.linewidth": 1.4, "lines.markersize": 4,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300,
        "text.usetex": False, "mathtext.fontset": "cm",
    })
