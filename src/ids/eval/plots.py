"""Shared figure writers. Fixed categorical order and one-hue sequential ramp across all reports."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]    # categorical slots, assigned in this order
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5",
                                                "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _hide_spines(ax) -> None:
    for s in ax.spines.values():
        s.set_visible(False)


def fig_matrix(M, windows, path, title) -> None:
    """Train x eval heatmap; M from Evaluator.matrix."""
    fig, ax = plt.subplots(figsize=(6.2, 0.75 * len(M.index) + 1.6))
    im = ax.imshow(M.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=SURFACE if v > 0.6 else INK)
    ax.set_xticks(range(M.shape[1]), [windows[w] for w in M.columns])
    ax.set_yticks(range(M.shape[0]), [f"through {windows[t]}" for t in M.index])
    ax.set_xlabel("evaluated on test set of window"); ax.set_title(title, loc="left")
    _hide_spines(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("macro-F1", color=INK2); cb.outline.set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_curves(series_by_label: dict, windows, path, title, ylabel="macro-F1 on that window's test set") -> None:
    """Lines over the windows. A label may carry a (color slot, dashed?) style via a tuple key."""
    x = np.arange(len(windows))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, (label, s) in enumerate(series_by_label.items()):
        slot, dashed = label[1] if isinstance(label, tuple) else (i, False)
        name = label[0] if isinstance(label, tuple) else label
        ax.plot(x, s.reindex(x).values, color=SERIES[slot], lw=2, ls="--" if dashed else "-", marker="o", ms=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        ax.annotate(f"{s.iloc[-1]:.2f}", (x[-1], s.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK2)
    ax.set_xticks(x, windows); ax.set_ylim(0, 1.05); ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(title, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_confusions(panels: list[tuple[np.ndarray, str]], names, path, xlabel="predicted") -> None:
    """Row-normalized confusion matrices as small multiples."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), squeeze=False)
    for ax, (cm, title) in zip(axes[0], panels):
        cm = cm.astype(float)
        norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                if cm[i].sum() and norm[i, j] >= 0.01:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color=SURFACE if norm[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel(xlabel); ax.set_ylabel("true")
        ax.set_title(title, loc="left")
        _hide_spines(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_per_class(pc_by_window: dict, names, windows, path, title) -> None:
    """Per-class F1 of the current model at each window; blank = class not present yet."""
    a = np.array([[pc_by_window.get(w, {}).get(c, np.nan) for w in range(len(windows))] for c in names])
    fig, ax = plt.subplots(figsize=(6.2, 0.45 * len(names) + 1.4))
    im = ax.imshow(a, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if not np.isnan(a[i, j]):
                ax.text(j, i, f"{a[i, j]:.2f}", ha="center", va="center", fontsize=8, color=SURFACE if a[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(windows)), windows); ax.set_yticks(range(len(names)), names)
    ax.set_title(title, loc="left"); _hide_spines(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("F1", color=INK2); cb.outline.set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
