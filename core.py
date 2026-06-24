"""
Вычислительное ядро для анализа обобщённого отображения параболы:
x_{n+1} = 1 - mu * |x_n|^z
Все тяжёлые функции векторизованы через NumPy: итерации по всем
значениям mu выполняются одновременно как операции над массивами.
"""
import csv
import math
import numpy as np

# ======================================================================
# Именованные константы
# ======================================================================
# Порог расхождения: итерация считается убежавшей, если |x| > OVERFLOW
OVERFLOW = 1e10
# Начальное условие для всех итерационных процедур
X0 = 0.1

# --- Бифуркационная диаграмма ---
N_MU = 1200          # Число точек по оси μ (разрешение диаграммы)
N_SKIP = 400         # Итераций на «прогрев» — отбрасываем переходный процесс
N_PLOT = 200         # Итераций для записи — столько точек рисуется на каждом μ

# --- Показатель Ляпунова ---
N_ITER_LYAPUNOV = 1000   # Итераций для усреднения ln|f'(x)|
N_SKIP_LYAPUNOV = 300    # Прогрев перед вычислением показателя

# --- Определение периода ---
N_SKIP_PERIOD = 5000     # Прогрев перед анализом периода (больше — надёжнее)
N_CHECK_PERIOD = 4096    # Длина орбиты для проверки периодичности
TOL_PERIOD = 1e-6        # Допуск: |x_{n} - x_{n+T}| < TOL => период T

# --- Бисекция точки бифуркации (оставлено для обратной совместимости) ---
BISECT_STEPS = 60
BISECT_TOL = 1e-14

# --- Грубый скан для поиска бифуркаций (оставлено для обратной совместимости) ---
N_SCAN = 2000

def generalized_map(x, mu, z):
    """Обобщённое отображение параболы (скаляр или массив)."""
    return 1.0 - mu * np.abs(x) ** z

# ======================================================================
# Бифуркационная диаграмма (векторизована)
# ======================================================================
def bifurcation_data(z, mu_min=0.0, mu_max=2.0, n_mu=N_MU,
                     n_skip=N_SKIP, n_plot=N_PLOT):
    """Возвращает (mu_values, x_values) для scatter-plot."""
    mu_arr = np.linspace(mu_min, mu_max, n_mu)
    x = np.full(n_mu, X0)
    for _ in range(n_skip):
        x = 1.0 - mu_arr * np.abs(x) ** z
        x = np.clip(x, -OVERFLOW, OVERFLOW)

    mu_out = np.empty(n_mu * n_plot)
    x_out = np.empty(n_mu * n_plot)
    for j in range(n_plot):
        x = 1.0 - mu_arr * np.abs(x) ** z
        x = np.clip(x, -OVERFLOW, OVERFLOW)
        mu_out[j * n_mu:(j + 1) * n_mu] = mu_arr
        x_out[j * n_mu:(j + 1) * n_mu] = x

    valid = np.isfinite(x_out) & (np.abs(x_out) < OVERFLOW)
    return mu_out[valid], x_out[valid]

# ======================================================================
# Показатель Ляпунова (векторизован)
# ======================================================================
def lyapunov_exponent(mu_arr, z, n_iter=N_ITER_LYAPUNOV, n_skip=N_SKIP_LYAPUNOV):
    """lambda(mu) = (1/N) * sum ln|f'(x_n)| для каждого mu."""
    mu_arr = np.atleast_1d(mu_arr).astype(float)
    n = len(mu_arr)
    x = np.full(n, X0)
    for _ in range(n_skip):
        x = 1.0 - mu_arr * np.abs(x) ** z
        x = np.clip(x, -OVERFLOW, OVERFLOW)

    lyap_sum = np.zeros(n)
    for _ in range(n_iter):
        df = np.abs(-mu_arr * z * np.abs(x) ** (z - 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            lyap_sum += np.where(df > 0, np.log(df), -50.0)
        x = 1.0 - mu_arr * np.abs(x) ** z

    result = lyap_sum / n_iter
    bad = ~np.isfinite(result)
    result[bad] = np.nan
    return result

# ======================================================================
# Паутинная диаграмма (скалярная)
# ======================================================================
def cobweb_data(mu, z, x0=X0, n_iter=80):
    """Возвращает (cobweb_x, cobweb_y) — ломаную для отрисовки."""
    cx = [x0, x0]
    cy = [0.0, 1.0 - mu * abs(x0) ** z]
    x = x0
    for _ in range(n_iter):
        y = 1.0 - mu * abs(x) ** z
        cx.extend([x, y])
        cy.extend([y, y])
        x = y
        if abs(x) > OVERFLOW:
            break
    return np.array(cx), np.array(cy)

# ======================================================================
# Определение периода — пакетная версия (векторизована)
# ======================================================================
def _detect_periods_batch(mu_arr, z, n_skip=N_SKIP_PERIOD, n_check=N_CHECK_PERIOD,
                          tol=TOL_PERIOD):
    """Определяет устойчивый период для каждого mu из массива."""
    mu_arr = np.asarray(mu_arr, dtype=float)
    n = len(mu_arr)
    x = np.full(n, X0)
    for _ in range(n_skip):
        x = 1.0 - mu_arr * np.abs(x) ** z

    orbit = np.empty((n_check, n))
    orbit[0] = x
    for i in range(1, n_check):
        orbit[i] = 1.0 - mu_arr * np.abs(orbit[i - 1]) ** z

    ref = orbit[-1]
    periods = np.zeros(n, dtype=int)

    for period in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if period * 8 > n_check:
            break

        undecided = periods == 0
        if not np.any(undecided):
            break

        match = np.ones(n, dtype=bool) & undecided
        for k in range(1, 8):
            diff = np.abs(orbit[-1 - k * period] - ref)
            match &= diff < tol

        if period > 1:
            sub_match = np.ones(n, dtype=bool) & match
            half = period // 2
            for k in range(1, 8):
                diff = np.abs(orbit[-1 - k * half] - ref)
                sub_match &= diff < tol
            match &= ~sub_match

        periods[match] = period

    diverged = ~np.isfinite(ref) | (np.abs(ref) > OVERFLOW)
    periods[diverged] = 0
    return periods

def _detect_period(mu, z, n_skip=N_SKIP_PERIOD, n_check=N_CHECK_PERIOD, tol=TOL_PERIOD):
    """Скалярная обёртка для совместимости."""
    return int(_detect_periods_batch(np.array([mu]), z, n_skip, n_check, tol)[0])

# ======================================================================
# Алгоритм Фейгенбаума: Суперустойчивые точки μ_n
# ======================================================================
def _compute_superstable_mu(z, n, mu_guess, tol=1e-15, max_iter=60):
    period = 2 ** n
    mu = mu_guess

    for _ in range(max_iter):
        x = 0.0
        dx_dmu = 0.0

        
        for _ in range(period):
            
            term = 1.0 - 2.0 * x * x
            x_new = mu * term

            
            dx_dmu = term + (-4.0 * mu * x) * dx_dmu

            x = x_new
            if abs(x) > 2.0:  
                break

        if abs(dx_dmu) < 1e-16:
            break

        
        step = -x / dx_dmu

        
        if abs(step) > 0.05:
            step = 0.05 * (1.0 if step >= 0 else -1.0)

        mu += step
        if abs(step) < tol:
            break

    return mu


def find_bifurcation_points(z, n_bifurcations=8):
    mu_values = []
    DELTA_FEIG = 4.6692016091029909

    
    guesses = [0.71, 0.81]

    for n in range(1, n_bifurcations + 1):
        if n <= len(guesses):
            guess = guesses[n - 1]
        else:
            
            diff = mu_values[-1] - mu_values[-2]
            guess = mu_values[-1] + diff / DELTA_FEIG

        mu_n = _compute_superstable_mu(None, n, guess)
        mu_values.append(mu_n)

    return np.array(mu_values)

# ======================================================================
# Константы Фейгенбаума
# ======================================================================
def feigenbaum_deltas(bif_points):
    """delta_n = (mu_n - mu_{n-1}) / (mu_{n+1} - mu_n)"""
    if len(bif_points) < 3:
        return np.array([])
    bp = np.asarray(bif_points)
    num = bp[1:-1] - bp[:-2]
    den = bp[2:] - bp[1:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(den) > 1e-15, num / den, np.nan)

def feigenbaum_alphas(z, bif_points):
    
    if len(bif_points) < 3:
        return np.array([])
    alphas = []
    for i in range(len(bif_points) - 1):
        period = 1 << (i + 1)
        mu = bif_points[i]
        x = 0.0
        for _ in range(period // 2):
            x = 1.0 - mu * abs(x) ** z
        d_curr = abs(x)

        mu_next = bif_points[i + 1]
        x = 0.0
        next_period = 1 << (i + 2)
        for _ in range(next_period // 2):
            x = 1.0 - mu_next * abs(x) ** z
        d_next = abs(x)

        if abs(d_next) > 1e-15:
            alphas.append(d_curr / d_next)
        else:
            alphas.append(np.nan)

    return np.array(alphas)

def feigenbaum_constants_vs_z(z_values, n_bif=6):
    
    delta_arr = np.full(len(z_values), np.nan)
    alpha_arr = np.full(len(z_values), np.nan)
    for i, z in enumerate(z_values):
        bp = find_bifurcation_points(z, n_bif)
        if len(bp) >= 4:
            ds = feigenbaum_deltas(bp)
            valid = ds[~np.isnan(ds)]
            if len(valid) > 0:
                delta_arr[i] = valid[-1]
            als = feigenbaum_alphas(z, bp)
            valid_a = als[~np.isnan(als)]
            if len(valid_a) > 0:
                alpha_arr[i] = valid_a[-1]

    return delta_arr, alpha_arr

# ======================================================================
# Скейлинг
# ======================================================================
def scaling_zoom_regions(z, n_zooms=3):
    
    bp = find_bifurcation_points(z, n_zooms + 3)
    if len(bp) < 3:
        return [(0.0, 2.0, -1.5, 1.5)]
    regions = [(0.0, bp[-1] * 1.05, -1.5, 1.5)]

    for k in range(min(n_zooms - 1, len(bp) - 2)):
        idx = k + 1
        if idx + 1 >= len(bp):
            break
        mu_width = bp[idx + 1] - bp[idx - 1]
        mu_lo = bp[idx] - mu_width * 0.3
        mu_hi = bp[idx + 1] + mu_width * 0.1

        _, x_data = bifurcation_data(z, mu_lo, mu_hi,
                                     n_mu=400, n_skip=500, n_plot=150)
        if len(x_data) > 0:
            x_lo = np.percentile(x_data, 1) - 0.05
            x_hi = np.percentile(x_data, 99) + 0.05
        else:
            x_lo, x_hi = -1.0, 1.0

        regions.append((mu_lo, mu_hi, x_lo, x_hi))

    return regions

# ======================================================================
# Экспорт таблицы Фейгенбаума в CSV
# ======================================================================
def export_feigenbaum_table(z, filename, n_bifurcations=8):
    
    bp = find_bifurcation_points(z, n_bifurcations)
    ds = feigenbaum_deltas(bp)
    als = feigenbaum_alphas(z, bp)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "period", "mu_n", "delta_n", "alpha_n"])

        for i in range(len(bp)):
            period = 2 ** (i + 1)
            d_val = ds[i - 1] if 0 <= i - 1 < len(ds) else ""
            a_val = als[i - 1] if 0 <= i - 1 < len(als) else ""

            writer.writerow([
                i + 1,
                period,
                f"{bp[i]:.12f}",
                f"{d_val:.8f}" if d_val != "" and not np.isnan(d_val) else "",
                f"{a_val:.8f}" if a_val != "" and not np.isnan(a_val) else "",
            ])

        if len(ds) >= 3:
            writer.writerow([])
            writer.writerow(["# Сходимость δ_n к пределу:"])
            for i, d in enumerate(ds):
                if not np.isnan(d):
                    writer.writerow([f"# δ_{i + 1}", f"{d:.8f}"])

    return bp, ds, als
