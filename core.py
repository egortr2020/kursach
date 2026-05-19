"""
Вычислительное ядро для анализа обобщённого отображения параболы:
    x_{n+1} = 1 - mu * |x_n|^z

Все тяжёлые функции векторизованы через NumPy: итерации по всем
значениям mu выполняются одновременно как операции над массивами.
"""

import csv
import numpy as np

# ======================================================================
#  Именованные константы (все «магические числа» собраны здесь)
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

# --- Бисекция точки бифуркации ---
BISECT_STEPS = 60        # Число шагов бисекции (даёт точность ~2^{-60} ≈ 1e-18)
BISECT_TOL = 1e-14       # Остановка бисекции по ширине интервала

# --- Грубый скан для поиска бифуркаций ---
N_SCAN = 2000            # Число точек μ в первичном скане


def generalized_map(x, mu, z):
    """Обобщённое отображение параболы (скаляр или массив)."""
    return 1.0 - mu * np.abs(x) ** z


# ======================================================================
#  Бифуркационная диаграмма (векторизована)
# ======================================================================

def bifurcation_data(z, mu_min=0.0, mu_max=2.0, n_mu=N_MU,
                     n_skip=N_SKIP, n_plot=N_PLOT):
    """
    Возвращает (mu_values, x_values) для scatter-plot.
    Все n_mu значений mu итерируются одновременно.
    """
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
#  Показатель Ляпунова (векторизован)
# ======================================================================

def lyapunov_exponent(mu_arr, z, n_iter=N_ITER_LYAPUNOV, n_skip=N_SKIP_LYAPUNOV):
    """
    lambda(mu) = (1/N) * sum ln|f'(x_n)|  для каждого mu.
    """
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
#  Паутинная диаграмма (скалярная — быстрая сама по себе)
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
#  Определение периода — пакетная версия (векторизована)
# ======================================================================

def _detect_periods_batch(mu_arr, z, n_skip=N_SKIP_PERIOD, n_check=N_CHECK_PERIOD,
                          tol=TOL_PERIOD):
    """
    Определяет устойчивый период для каждого mu из массива.
    Все mu итерируются одновременно. Возвращает int-массив периодов
    (0 = не определён / хаос).
    """
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


def _detect_period(mu, z, n_skip=N_SKIP_PERIOD, n_check=N_CHECK_PERIOD,
                   tol=TOL_PERIOD):
    """Скалярная обёртка для совместимости (используется в бисекции)."""
    return int(_detect_periods_batch(np.array([mu]), z, n_skip, n_check, tol)[0])


# ======================================================================
#  Поиск точек бифуркаций
# ======================================================================

def _iterate_fp(mu, z, period, x_start):
    """
    Вычисляет f^period(x_start) и мультипликатор (произведение производных).
    Возвращает (f^p(x), multiplier).  При расхождении возвращает (nan, nan).
    """
    y = x_start
    mult = 1.0
    for _ in range(period):
        if abs(y) > OVERFLOW:
            return np.nan, np.nan
        if abs(y) > 1e-30:
            mult *= -mu * z * abs(y) ** (z - 1) * np.sign(y)
        else:
            mult = 0.0
        y = 1.0 - mu * abs(y) ** z
        if not np.isfinite(y):
            return np.nan, np.nan
    return y, mult


def _newton_find_cycle(mu, z, period, x_start, max_iter=40):
    """
    Методом Ньютона находит неподвижную точку f^period (точку p-цикла).
    Возвращает (x_fixed, multiplier).
    """
    x = x_start
    for _ in range(max_iter):
        y, mult = _iterate_fp(mu, z, period, x)
        if not np.isfinite(y) or not np.isfinite(mult):
            break
        g = y - x
        denom = mult - 1.0
        if abs(denom) < 1e-30:
            break
        x -= g / denom
        if abs(g) < 1e-15:
            break
    # Финальное вычисление мультипликатора в уточнённой точке
    _, mult = _iterate_fp(mu, z, period, x)
    return x, mult


def _bisect_bifurcation(z, p_before, mu_lo, mu_hi):
    """
    Уточняет точку бифуркации бисекцией по устойчивости p-цикла.
    Использует метод Ньютона для нахождения точной точки p-цикла
    и проверяет знак |мультипликатор| − 1.
    """
    # Находим p-цикл при mu_lo (он здесь заведомо устойчив)
    n_warmup = max(N_SKIP_PERIOD, p_before * 50)
    x = X0
    for _ in range(n_warmup):
        x = 1.0 - mu_lo * abs(x) ** z
    x_fp, _ = _newton_find_cycle(mu_lo, z, p_before, x)

    # Верификация: расширяем скобку, пока mu_hi действительно
    # не окажется по ту сторону бифуркации (|mult| >= 1)
    x_track = x_fp
    step = mu_hi - mu_lo
    x_fp_hi, mult_hi = _newton_find_cycle(mu_hi, z, p_before, x_track)
    while abs(mult_hi) < 1.0 and mu_hi < 2.0:
        x_track = x_fp_hi
        mu_hi = min(2.0, mu_hi + step)
        x_fp_hi, mult_hi = _newton_find_cycle(mu_hi, z, p_before, x_track)

    lo, hi = mu_lo, mu_hi
    for _ in range(BISECT_STEPS):
        mid = (lo + hi) / 2.0
        x_fp_mid, mult = _newton_find_cycle(mid, z, p_before, x_fp)
        if np.isfinite(mult) and abs(mult) < 1.0:
            lo = mid
            x_fp = x_fp_mid   # Отслеживаем p-цикл вдоль устойчивой ветви
        else:
            hi = mid           # Расходимость или |mult|>=1 → за бифуркацией
        if hi - lo < BISECT_TOL:
            break
    return (lo + hi) / 2.0


def find_bifurcation_points(z, n_bifurcations=8):
    """
    Находит точки бифуркаций удвоения периода.
    Грубый скан (векторизованный) + предсказание + бисекция.
    """
    n_scan = N_SCAN
    scan_mus = np.linspace(0.001, 2.0, n_scan)
    periods = _detect_periods_batch(scan_mus, z)

    stable_zones = []
    run_start = 0
    for i in range(1, n_scan):
        if periods[i] != periods[run_start] or i == n_scan - 1:
            # Если это последний элемент и он принадлежит текущей серии,
            # включаем его в длину
            end = i if periods[i] != periods[run_start] else i + 1
            length = end - run_start
            if length >= 3 and periods[run_start] > 0:
                stable_zones.append((
                    int(periods[run_start]),
                    scan_mus[run_start],
                    scan_mus[min(end - 1, n_scan - 1)],
                ))
            run_start = i

    bif_points = []
    for zi in range(len(stable_zones) - 1):
        p1, _, mu1_end = stable_zones[zi]
        p2, mu2_start, _ = stable_zones[zi + 1]
        if p2 == p1 * 2:
            bif_mu = _bisect_bifurcation(z, p1, mu1_end, mu2_start)
            bif_points.append(bif_mu)

    while len(bif_points) >= 2 and len(bif_points) < n_bifurcations:
        n = len(bif_points)
        d_last = bif_points[-1] - bif_points[-2]
        ds = feigenbaum_deltas(np.array(bif_points))
        delta_est = ds[-1] if len(ds) > 0 and np.isfinite(ds[-1]) else 4.5
        delta_est = max(2.0, min(delta_est, 20.0))

        d_next = d_last / delta_est
        predicted = bif_points[-1] + d_next
        if predicted > 2.0 or predicted < bif_points[-1]:
            break

        # Используем Ньютон-трекинг вместо определения периода:
        # p_before — текущий период, его цикл рождается при bif_points[-1]
        # и теряет устойчивость при следующей бифуркации.
        # mu_lo берём в середине предсказанного окна (цикл хорошо устойчив),
        # mu_hi — с запасом за предсказание.
        p_before = 1 << n
        mu_lo = bif_points[-1] + d_next * 0.4
        mu_hi = min(2.0, predicted + d_next * 3)

        bif_mu = _bisect_bifurcation(z, p_before, mu_lo, mu_hi)
        if bif_mu > bif_points[-1] + 1e-15:
            bif_points.append(bif_mu)
        else:
            break

    return np.array(bif_points)


# ======================================================================
#  Константы Фейгенбаума
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
    """
    alpha_n через отношение расстояний суперустойчивых орбит
    от критической точки x=0.
    """
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
    """Предельные delta и alpha для набора значений z."""
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
#  Скейлинг
# ======================================================================

def scaling_zoom_regions(z, n_zooms=3):
    """Прямоугольные области для демонстрации самоподобия."""
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
#  Экспорт таблицы Фейгенбаума в CSV
# ======================================================================

def export_feigenbaum_table(z, filename, n_bifurcations=8):
    """
    Сохраняет CSV-таблицу с колонками: n, mu_n, delta_n, alpha_n.
    Возвращает (bif_points, deltas, alphas) для дальнейшего использования.
    """
    bp = find_bifurcation_points(z, n_bifurcations)
    ds = feigenbaum_deltas(bp)
    als = feigenbaum_alphas(z, bp)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "mu_n", "delta_n", "alpha_n"])
        for i in range(len(bp)):
            d = ds[i - 1] if 0 <= i - 1 < len(ds) else ""
            a = als[i - 1] if 0 <= i - 1 < len(als) else ""
            writer.writerow([i + 1, f"{bp[i]:.12f}",
                             f"{d:.8f}" if d != "" else "",
                             f"{a:.8f}" if a != "" else ""])

    return bp, ds, als
