#!/usr/bin/env python3
"""
Генерация CSV-таблиц для курсовой работы.
Сохраняет в папку tables/.

Для каждого z генерируется подробная таблица, показывающая
пошаговую сходимость δ_n и α_n к предельным значениям.

Запуск:  python export_tables.py
"""

import csv
import os
import numpy as np

import core

OUTDIR = "tables"

Z_VALUES = [2.0, 2.1, 2.2, 2.3]

N_BIF = 8

REFERENCE = {
    2.0: {"delta": 4.6692016091029, "alpha": 2.5029078750959},
}


def save_convergence_table(z, outdir):
    bp = core.find_bifurcation_points(z, n_bifurcations=N_BIF)
    ds = core.feigenbaum_deltas(bp)
    als = core.feigenbaum_alphas(z, bp)

    filename = os.path.join(outdir, f"convergence_z{z:.2f}.csv")
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n",
            "период (2^n)",
            "mu_n",
            "Delta_mu_n = mu_n - mu_{n-1}",
            "delta_n = Delta_mu_{n-1} / Delta_mu_n",
            "alpha_n",
        ])

        for i in range(len(bp)):
            period = 2 ** (i + 1)
            mu = bp[i]

            # Δμ = μ_n - μ_{n-1}
            if i >= 1:
                d_mu = bp[i] - bp[i - 1]
                d_mu_str = f"{d_mu:.12f}"
            else:
                d_mu_str = ""

            # δ_n = Δμ_{n-1} / Δμ_n  (начинается с n=1, т.е. i=2)
            d_idx = i - 2
            if 0 <= d_idx < len(ds):
                delta_str = f"{ds[d_idx]:.8f}"
            else:
                delta_str = ""

            # α_n
            a_idx = i - 1
            if 0 <= a_idx < len(als):
                alpha_str = f"{als[a_idx]:.8f}"
            else:
                alpha_str = ""

            writer.writerow([
                i + 1,
                period,
                f"{mu:.12f}",
                d_mu_str,
                delta_str,
                alpha_str,
            ])

        
        ref = REFERENCE.get(z, {})
        delta_limit = ds[-1] if len(ds) > 0 else ""
        alpha_limit = als[-1] if len(als) > 0 else ""
        writer.writerow([])
        writer.writerow([
            "предел",
            "",
            "",
            "",
            f"{delta_limit:.8f}" if delta_limit != "" else "не определён",
            f"{alpha_limit:.8f}" if alpha_limit != "" else "не определён",
        ])
        if ref:
            writer.writerow([
                "эталон (лит.)",
                "",
                "",
                "",
                f"{ref['delta']:.10f}",
                f"{ref['alpha']:.10f}",
            ])
            if delta_limit != "" and np.isfinite(delta_limit):
                d_err = abs(delta_limit - ref["delta"]) / ref["delta"] * 100
                a_err = abs(alpha_limit - ref["alpha"]) / ref["alpha"] * 100 if alpha_limit != "" and np.isfinite(alpha_limit) else float("nan")
                writer.writerow([
                    "отклонение %",
                    "",
                    "",
                    "",
                    f"{d_err:.4f}%",
                    f"{a_err:.4f}%" if np.isfinite(a_err) else "",
                ])

    print(f"  -> {filename}")


def save_summary_table(outdir):
    """
    Сводная таблица: предельные δ(z) и α(z) для всех z.
    """
    z_arr = np.array(Z_VALUES)
    d_arr, a_arr = core.feigenbaum_constants_vs_z(z_arr, n_bif=N_BIF)

    filename = os.path.join(outdir, "summary_delta_alpha_vs_z.csv")
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "z",
            "delta (вычислено)",
            "alpha (вычислено)",
            "delta (эталон)",
            "alpha (эталон)",
            "delta отклонение %",
            "alpha отклонение %",
        ])
        for i, z in enumerate(z_arr):
            d = d_arr[i]
            a = a_arr[i]
            ref = REFERENCE.get(z, {})
            d_ref = ref.get("delta", "")
            a_ref = ref.get("alpha", "")

            d_err = ""
            a_err = ""
            if d_ref != "" and np.isfinite(d):
                d_err = f"{abs(d - d_ref) / d_ref * 100:.4f}"
            if a_ref != "" and np.isfinite(a):
                a_err = f"{abs(a - a_ref) / a_ref * 100:.4f}"

            writer.writerow([
                f"{z:.2f}",
                f"{d:.8f}" if np.isfinite(d) else "",
                f"{a:.8f}" if np.isfinite(a) else "",
                f"{d_ref:.10f}" if d_ref != "" else "",
                f"{a_ref:.10f}" if a_ref != "" else "",
                d_err,
                a_err,
            ])

    print(f"  -> {filename}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    for z in Z_VALUES:
        print(f"[z = {z:.2f}] Таблица сходимости...")
        save_convergence_table(z, OUTDIR)

    print("Сводная таблица delta(z), alpha(z)...")
    save_summary_table(OUTDIR)

    print(f"\nГотово! Все таблицы сохранены в {os.path.abspath(OUTDIR)}/")


if __name__ == "__main__":
    main()
