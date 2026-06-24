#!/usr/bin/env python3
"""
Генерация всех ключевых графиков в PNG (300 dpi) для курсовой работы.
Сохраняет в папку figures/.

Запуск:  python export_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core

OUTDIR = "figures"
DPI = 300

Z_VALUES = [2.0, 2.1, 2.2, 2.3]

COLORS = {
    "bg": "#1e1e2e",
    "panel": "#2a2a3d",
    "text": "#cdd6f4",
    "accent1": "#89b4fa",
    "accent2": "#f38ba8",
    "accent3": "#a6e3a1",
    "accent4": "#fab387",
    "accent5": "#cba6f7",
    "grid": "#45475a",
    "scatter": "#89dceb",
}

def _style_ax(ax):
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors=COLORS["text"], labelsize=9)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.grid(True, color=COLORS["grid"], alpha=0.3, linewidth=0.5)

def save_bifurcation(z, outdir):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS["bg"])
    _style_ax(ax)

    mu_data, x_data = core.bifurcation_data(z, mu_min=0.0, mu_max=2.0,
                                             n_mu=1500, n_skip=500, n_plot=250)
    ax.scatter(mu_data, x_data, s=0.01, c=COLORS["scatter"], alpha=0.5, linewidths=0)
    ax.set_xlim(0, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("$\\mu$", fontsize=12)
    ax.set_ylabel("$x$", fontsize=12)
    ax.set_title(f"Бифуркационная диаграмма  $x_{{n+1}} = 1 - \\mu \\cdot |x_n|^{{{z:.1f}}}$",
                 fontsize=13)
    fig.savefig(os.path.join(outdir, f"bifurcation_z{z:.1f}.png"),
                dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_bifurcation_zooms(z, outdir):
    """
    Увеличенные фрагменты бифуркационного дерева вблизи точки накопления.
    3 уровня зума: каждый следующий — окрестность последних бифуркаций.
    Нужны для чтения точек μ_n и демонстрации самоподобия.
    """
    bp = core.find_bifurcation_points(z, n_bifurcations=8)
    if len(bp) < 3:
        return

    mu_lo1 = bp[0] - (bp[1] - bp[0]) * 0.3
    mu_hi1 = bp[-1] + (bp[-1] - bp[-2]) * 0.5
    mu_hi1 = min(mu_hi1, 2.0)

    
    if len(bp) >= 4:
        mu_lo2 = bp[-4] - (bp[-3] - bp[-4]) * 0.3
    else:
        mu_lo2 = bp[-2] - (bp[-1] - bp[-2]) * 0.3
    mu_hi2 = bp[-1] + (bp[-1] - bp[-2]) * 0.3
    mu_hi2 = min(mu_hi2, 2.0)

    
    mu_lo3 = bp[-2] - (bp[-1] - bp[-2]) * 0.3
    mu_hi3 = bp[-1] + (bp[-1] - bp[-2]) * 0.2
    mu_hi3 = min(mu_hi3, 2.0)

    zoom_specs = [
        (mu_lo1, mu_hi1, "Зум 1: все бифуркации"),
        (mu_lo2, mu_hi2, "Зум 2: последние бифуркации"),
        (mu_lo3, mu_hi3, "Зум 3: окрестность точки накопления"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=COLORS["bg"])

    for k, (ax, (mu_lo, mu_hi, title)) in enumerate(zip(axes, zoom_specs)):
        _style_ax(ax)
        n_mu = 1200 if k == 0 else 1000
        mu_data, x_data = core.bifurcation_data(z, mu_min=mu_lo, mu_max=mu_hi,
                                                 n_mu=n_mu, n_skip=600, n_plot=200)
        ax.scatter(mu_data, x_data, s=0.04, c=COLORS["scatter"],
                   alpha=0.5, linewidths=0)

        
        if len(x_data) > 0:
            x_lo = np.percentile(x_data, 0.5) - 0.05
            x_hi = np.percentile(x_data, 99.5) + 0.05
        else:
            x_lo, x_hi = -1.5, 1.5

        ax.set_xlim(mu_lo, mu_hi)
        ax.set_ylim(x_lo, x_hi)
        ax.set_xlabel("$\\mu$", fontsize=10)
        if k == 0:
            ax.set_ylabel("$x$", fontsize=10)
        ax.set_title(f"{title}  ($z = {z:.1f}$)", fontsize=11)

        
        for mu_bif in bp:
            if mu_lo <= mu_bif <= mu_hi:
                ax.axvline(mu_bif, color=COLORS["accent2"], linewidth=0.6,
                           alpha=0.5, linestyle=":")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"bifurcation_zooms_z{z:.1f}.png"),
                dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_lyapunov(z, outdir):
    """Показатель Ляпунова."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLORS["bg"])
    _style_ax(ax)

    mu_arr = np.linspace(0.01, 2.0, 1200)
    lyap = core.lyapunov_exponent(mu_arr, z, n_iter=1000, n_skip=400)

    pos = lyap >= 0
    neg = lyap < 0
    ax.fill_between(mu_arr, lyap, 0, where=pos, color=COLORS["accent2"],
                    alpha=0.35, label="$\\lambda > 0$ (хаос)")
    ax.fill_between(mu_arr, lyap, 0, where=neg, color=COLORS["accent3"],
                    alpha=0.35, label="$\\lambda < 0$ (порядок)")
    ax.plot(mu_arr, lyap, color=COLORS["accent1"], linewidth=0.8)
    ax.axhline(0, color=COLORS["text"], linewidth=1, linestyle="--", alpha=0.6)

    ax.set_xlim(0, 2)
    ax.set_ylim(min(-3, np.nanmin(lyap) - 0.3), max(1.5, np.nanmax(lyap) + 0.3))
    ax.set_xlabel("$\\mu$", fontsize=12)
    ax.set_ylabel("$\\lambda$", fontsize=12)
    ax.set_title(f"Показатель Ляпунова  $z = {z:.1f}$", fontsize=13)
    ax.legend(loc="lower left", fontsize=9, facecolor=COLORS["panel"],
              edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    fig.savefig(os.path.join(outdir, f"lyapunov_z{z:.1f}.png"),
                dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_scaling(z, outdir):
    """Увеличенные фрагменты (скейлинг / самоподобие)."""
    regions = core.scaling_zoom_regions(z, n_zooms=3)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=COLORS["bg"])
    titles = ["Полный вид", "Зум $\\times$1", "Зум $\\times$2"]

    for k, ax in enumerate(axes):
        _style_ax(ax)
        if k < len(regions):
            mu_lo, mu_hi, x_lo, x_hi = regions[k]
        else:
            mu_lo, mu_hi, x_lo, x_hi = 0.0, 2.0, -1.5, 1.5

        n_mu = 800 if k == 0 else 600
        mu_data, x_data = core.bifurcation_data(z, mu_min=mu_lo, mu_max=mu_hi,
                                                 n_mu=n_mu, n_skip=500, n_plot=200)
        ax.scatter(mu_data, x_data, s=0.05, c=COLORS["scatter"],
                   alpha=0.5, linewidths=0)
        ax.set_xlim(mu_lo, mu_hi)
        ax.set_ylim(x_lo, x_hi)
        ax.set_xlabel("$\\mu$", fontsize=10)
        if k == 0:
            ax.set_ylabel("$x$", fontsize=10)
        ax.set_title(f"{titles[k]}  ($z = {z:.1f}$)", fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"scaling_z{z:.1f}.png"),
                dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_feigenbaum_convergence(z, outdir):
    """
    Сходимость δ_n и α_n для конкретного z.
    """
    bp = core.find_bifurcation_points(z, n_bifurcations=8)
    ds = core.feigenbaum_deltas(bp)
    als = core.feigenbaum_alphas(z, bp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=COLORS["bg"])

    # --- δ_n ---
    _style_ax(ax1)
    if len(ds) > 0:
        ns = np.arange(1, len(ds) + 1)
        ax1.plot(ns, ds, "o-", color=COLORS["accent1"], markersize=8,
                 linewidth=2, label="$\\delta_n$", zorder=5)
        
        for i, (n, d) in enumerate(zip(ns, ds)):
            ax1.annotate(f"{d:.3f}", (n, d), textcoords="offset points",
                         xytext=(0, 12), ha="center", fontsize=8,
                         color=COLORS["text"])
        
        ax1.axhline(ds[-1], color=COLORS["accent1"], linestyle=":", alpha=0.5,
                     label=f"предел: {ds[-1]:.4f}")

    ax1.set_xlabel("$n$", fontsize=11)
    ax1.set_ylabel("$\\delta_n$", fontsize=11)
    ax1.set_title(f"Сходимость $\\delta_n$  ($z = {z:.1f}$)", fontsize=12)
    ax1.legend(loc="best", fontsize=9, facecolor=COLORS["panel"],
               edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    # --- α_n ---
    _style_ax(ax2)
    if len(als) > 0:
        ns_a = np.arange(1, len(als) + 1)
        ax2.plot(ns_a, als, "s-", color=COLORS["accent2"], markersize=8,
                 linewidth=2, label="$\\alpha_n$", zorder=5)
        for i, (n, a) in enumerate(zip(ns_a, als)):
            if np.isfinite(a):
                ax2.annotate(f"{a:.3f}", (n, a), textcoords="offset points",
                             xytext=(0, 12), ha="center", fontsize=8,
                             color=COLORS["text"])
        ax2.axhline(als[-1], color=COLORS["accent2"], linestyle=":", alpha=0.5,
                     label=f"предел: {als[-1]:.4f}")

    ax2.set_xlabel("$n$", fontsize=11)
    ax2.set_ylabel("$\\alpha_n$", fontsize=11)
    ax2.set_title(f"Сходимость $\\alpha_n$  ($z = {z:.1f}$)", fontsize=12)
    ax2.legend(loc="best", fontsize=9, facecolor=COLORS["panel"],
               edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"feigenbaum_z{z:.1f}.png"),
                dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_feigenbaum_vs_z(outdir):
    """
    Зависимость δ(z) и α(z)
    """
    z_arr = np.arange(2.0, 4.05, 0.1)
    d_arr, a_arr = core.feigenbaum_constants_vs_z(z_arr, n_bif=6)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=COLORS["bg"])
    _style_ax(ax)

    mask_d = ~np.isnan(d_arr)
    mask_a = ~np.isnan(a_arr)
    if np.any(mask_d):
        ax.plot(z_arr[mask_d], d_arr[mask_d], "o-", color=COLORS["accent1"],
                markersize=5, linewidth=2, label="$\\delta(z)$")
    if np.any(mask_a):
        ax.plot(z_arr[mask_a], a_arr[mask_a], "s-", color=COLORS["accent2"],
                markersize=5, linewidth=2, label="$\\alpha(z)$")

    ax.axhline(4.6692, color=COLORS["accent1"], linestyle="--", alpha=0.4,
               label="$\\delta = 4.6692$ (эталон $z=2$)")
    ax.axhline(2.5029, color=COLORS["accent2"], linestyle="--", alpha=0.4,
               label="$\\alpha = 2.5029$ (эталон $z=2$)")

    ax.set_xlabel("$z$ (порядок экстремума)", fontsize=11)
    ax.set_ylabel("Значение константы", fontsize=11)
    ax.set_title("Зависимость констант Фейгенбаума от $z$", fontsize=12)
    ax.legend(loc="best", fontsize=8, facecolor=COLORS["panel"],
              edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    fig.savefig(os.path.join(outdir, "feigenbaum_vs_z.png"),
                dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    total = len(Z_VALUES) * 5 + 1
    done = 0

    for z in Z_VALUES:
        print(f"[z = {z:.1f}] Бифуркационная диаграмма...")
        save_bifurcation(z, OUTDIR)
        done += 1
        print(f"  ({done}/{total})")

        print(f"[z = {z:.1f}] Увеличенные фрагменты (зумы бифуркаций)...")
        save_bifurcation_zooms(z, OUTDIR)
        done += 1
        print(f"  ({done}/{total})")

        print(f"[z = {z:.1f}] Показатель Ляпунова...")
        save_lyapunov(z, OUTDIR)
        done += 1
        print(f"  ({done}/{total})")

        print(f"[z = {z:.1f}] Скейлинг...")
        save_scaling(z, OUTDIR)
        done += 1
        print(f"  ({done}/{total})")

        print(f"[z = {z:.1f}] Сходимость delta_n, alpha_n...")
        save_feigenbaum_convergence(z, OUTDIR)
        done += 1
        print(f"  ({done}/{total})")

    print("Зависимость констант Фейгенбаума от z (плотная сетка)...")
    save_feigenbaum_vs_z(OUTDIR)
    done += 1
    print(f"  ({done}/{total})")

    print(f"\nГотово! Все графики сохранены в {os.path.abspath(OUTDIR)}/")


if __name__ == "__main__":
    main()
