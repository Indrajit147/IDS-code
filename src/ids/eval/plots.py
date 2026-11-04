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
    ends = sorted((s.iloc[-1], i) for i, s in enumerate(series_by_label.values()))
    label_y, floor = {}, -1.0
    for v, i in ends:                                    # keep end labels at least one line apart
        floor = max(v, floor + 0.055)
        label_y[i] = floor
    for i, (label, s) in enumerate(series_by_label.items()):
        slot, dashed = label[1] if isinstance(label, tuple) else (i, False)
        name = label[0] if isinstance(label, tuple) else label
        ax.plot(x, s.reindex(x).values, color=SERIES[slot], lw=2, ls="--" if dashed else "-", marker="o", ms=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        ax.annotate(f"{s.iloc[-1]:.2f}", (x[-1], label_y[i]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK2, annotation_clip=False)
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


def fig_signal_traces(traces: dict, windows, signals, path) -> None:
    """Pooled stream at each boundary: rolling mean of each signal, with detections as ticks."""
    n = len(traces)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.3), sharey=True, squeeze=False)
    for ax, (d, tr) in zip(axes[0], traces.items()):
        total = len(next(iter(tr["signals"].values())))
        win = max(total // 60, 10)
        kernel = np.ones(win) / win
        x = np.arange(win - 1, total)
        for i, sig in enumerate(signals):
            ax.plot(x, np.convolve(tr["signals"][sig], kernel, mode="valid"), color=SERIES[i], lw=1.5, label=sig)
            fires = tr["fires"][sig]
            ax.plot(fires, [1.05 + 0.05 * i] * len(fires), ls="", marker="|", ms=8, mew=2, color=SERIES[i])
        ax.axvline(tr["n_pre"], color=INK2, lw=1, ls="--")
        ax.set_title(f"into {windows[d]}", loc="left")
        ax.set_xlabel("flows in stream"); ax.set_ylim(0, 1.18); ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 3))
    axes[0][0].set_ylabel("rolling mean of signal")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(signals), loc="upper right")
    fig.text(0.01, 0.95, "Dashed line: boundary. Ticks at the top of each panel: ADWIN detections.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(path, dpi=150); plt.close(fig)


def fig_escalation(front, chosen, path) -> None:
    """Escalation rate against tau_c, and the cost-benefit frontier, for a few values of tau_d."""
    shown = [t for t in (1.01, 0.2, 0.05) if t in set(front["tau_d"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
    for i, td in enumerate(shown):
        g = front[front["tau_d"] == td].sort_values("tau_c")
        label = "disagreement off" if td > 1 else f"τd = {td:g}"
        style = dict(color=SERIES[i], lw=2, label=label, ls="--" if td > 1 else "-", zorder=3 if td > 1 else 2)
        a.plot(g["tau_c"], g["escalation_rate"], **style)
        b.plot(g["escalation_rate"], g["error_capture"], **style)
    a.set_xlabel("τc (escalate if confidence is below this)"); a.set_ylabel("escalation rate")
    a.set_title("Escalation rate vs τ", loc="left"); a.legend(frameon=False, loc="upper left")
    b.set_xlabel("escalation rate"); b.set_ylabel("share of edge errors escalated")
    b.set_title("Cost and benefit", loc="left")
    b.plot([chosen["escalation_rate"]], [chosen["error_capture"]], ls="", marker="o", ms=9, color=INK,
           markeredgecolor=SURFACE, markeredgewidth=2)
    b.annotate(f"chosen: τc {chosen['tau_c']:.2f}, τd {chosen['tau_d']:g}", (chosen["escalation_rate"], chosen["error_capture"]),
               xytext=(55, -48), textcoords="offset points", color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    for ax in (a, b):
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_psi_by_window(shift, windows, alert, path) -> None:
    """Mean PSI per window, from the new traffic and from the control."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, (kind, label) in enumerate((("current", "new window vs running baseline"), ("control", "control: previous window"))):
        g = shift[shift["kind"] == kind].groupby("window")["psi"].mean()
        ax.plot(g.index, g.values, color=SERIES[i], lw=2, marker="o", ms=7, markeredgecolor=SURFACE,
                markeredgewidth=2, label=label)
    ax.axhline(alert, color=INK2, lw=1, ls="--")
    ax.annotate(f"PSI alert {alert:g}", (ax.get_xlim()[0], alert), xytext=(4, 4), textcoords="offset points", color=INK2)
    ids = sorted(shift["window"].unique())
    ax.set_xticks(ids, [windows[w] for w in ids]); ax.set_ylabel("mean PSI over clients and features")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_ylim(0, max(alert, shift.groupby(["kind", "window"])["psi"].mean().max()) * 1.45)
    ax.legend(frameon=False, loc="upper right"); ax.set_title("Distribution shift per window", loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
