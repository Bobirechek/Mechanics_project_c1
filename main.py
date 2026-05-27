import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

# 1. ФИЗИЧЕСКИЕ КОНСТАНТЫ
GM = 3.986_004_418e14        # гравитационный параметр Земли, м³/с²
R_EARTH = 6.371e6            # средний радиус Земли, м


# 2. МОДЕЛЬ АТМОСФЕРЫ: rho(h) = rho0 * exp(-(h - h_ref) / H)
RHO0 = 1.916e-11    # плотность при h_ref = 300 км, кг/м³
H_REF = 300.0e3     # опорная высота, м
H_ATM = 52.0e3      # масштаб высоты, м

def atm_density(h):
    """Плотность атмосферы [кг/м³] при высоте h [м]."""
    return RHO0 * np.exp(-(h - H_REF) / H_ATM)

# Контрольные значения COESA-76 (для проверки модели на графике)
H_CHECK = np.array([200, 250, 300, 350, 400, 450, 500]) * 1e3  # м
RHO_CHECK = np.array([2.54e-10, 6.07e-11, 1.916e-11, 7.01e-12,
                       2.803e-12, 1.23e-12, 5.21e-13])            # кг/м³

# 3. ПАРАМЕТРЫ СПУТНИКОВ — три набора входных данных
SATELLITES = [
    {'name': 'А', 'label': 'Спутник А  (β≈91 кг/м²)',
     'Cd': 2.2, 'S': 5.0,  'm': 1000, 'h0': 400e3,
     'color': '#1f78b4', 'ls': '-'},
    {'name': 'Б', 'label': 'Спутник Б  (β≈45 кг/м²)',
     'Cd': 2.2, 'S': 10.0, 'm': 1000, 'h0': 400e3,
     'color': '#e31a1c', 'ls': '--'},
    {'name': 'В', 'label': 'Спутник В  (β≈182 кг/м²)',
     'Cd': 2.2, 'S': 2.5,  'm': 1000, 'h0': 400e3,
     'color': '#33a02c', 'ls': '-.'},
]


# 4. АНАЛИТИЧЕСКАЯ ФОРМУЛА ВРЕМЕНИ СПУСКА
def analytic_time(h_start, h_end, Cd, S, m):
    a_start = R_EARTH + h_start
    a_end = R_EARTH + h_end
    a_avg = 0.5 * (a_start + a_end)
    B = Cd * S / m                     # обратный балл. коэф., м²/кг
    C = np.sqrt(GM * a_avg)            # sqrt(GM*a_avg), м²/с
    return (H_ATM / (B * RHO0 * C) *
            (np.exp((h_start - H_REF) / H_ATM) -
             np.exp((h_end   - H_REF) / H_ATM)))


# 5. РАЗНОСТНАЯ СХЕМА — ИТЕРАЦИЯ ПО ВИТКАМ
def orbit_iteration(Cd, S, m, h_start, h_stop=80e3, save_every=10):
    """
    Итерация по виткам — разностная алгебраическая схема.
    Один шаг = один орбитальный виток.
    Возвращает массивы: время [с], высота [м].
    """
    B = Cd * S / m                        # м²/кг
    a = R_EARTH + float(h_start)          # начальный радиус, м
    t = 0.0                               # начальное время, с
    t_arr = [t]
    h_arr = [a - R_EARTH]
    k = 0

    while (a - R_EARTH) > h_stop:
        rho_k = atm_density(a - R_EARTH)              # (1) плотность
        delta_a = -2.0 * np.pi * B * rho_k * a**2     # (2) изменение радиуса
        T_orb = 2.0 * np.pi * np.sqrt(a**3 / GM)      # (3) период витка
        a += delta_a                                  # (4) новый радиус
        t += T_orb                                    # (5) накопленное время
        k += 1
        if k % save_every == 0:
            t_arr.append(t)
            h_arr.append(a - R_EARTH)

    t_arr.append(t)
    h_arr.append(a - R_EARTH)
    return np.array(t_arr), np.array(h_arr)


# 6. ПОЛНАЯ 2D СИМУЛЯЦИЯ - РАЗНОСТНАЯ СХЕМА ЭЙЛЕРА (декарт. коорд.)
def simulate_2d(Cd, S, m, h_start, n_orbits=4, tau=10.0):
    """
    2D-симуляция методом Эйлера (начало — идеальная круговая орбита).
    Возвращает массивы координат x, y в метрах.
    """
    B = Cd * S / m
    r0 = R_EARTH + h_start
    vc = np.sqrt(GM / r0)
    t_end = n_orbits * 2.0 * np.pi * r0 / vc

    x, y, vx, vy = r0, 0.0, 0.0, vc
    x_arr, y_arr = [x], [y]
    t = 0.0
    step = 0

    while t < t_end:
        r = np.hypot(x, y)
        h = r - R_EARTH
        rho = atm_density(max(h, 0.0))
        v = np.hypot(vx, vy)

        # Ускорения (гравитация + аэроторможение)
        fg = -GM / r**3
        fd = -0.5 * B * rho * v
        ax = fg * x + fd * vx
        ay = fg * y + fd * vy

        # Шаг Эйлера
        x += vx * tau;   y  += vy * tau
        vx += ax * tau;   vy += ay * tau
        t += tau;  step += 1
        if step % 8 == 0:
            x_arr.append(x)
            y_arr.append(y)

    return np.array(x_arr), np.array(y_arr)


# 7. ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ
def time_at_altitude(t_arr, h_arr, target_m):
    for i, h in enumerate(h_arr):
        if h <= target_m:
            return t_arr[i]
    return t_arr[-1]


# 8. ВЫЧИСЛИТЕЛЬНЫЙ ЭКСПЕРИМЕНТ
print("=" * 65)
print("  ТЕМА 10: ДВИЖЕНИЕ СПУТНИКА В АТМОСФЕРЕ ЗЕМЛИ")
print("  Разностная схема по виткам + аналитическая формула")
print("=" * 65)

# --- Безразмерные параметры ---
T0 = np.sqrt(R_EARTH**3 / GM)
V0 = np.sqrt(GM / R_EARTH)
print(f"\n  Масштабные факторы (безразмерная форма):")
print(f"    L0 = R_Земли = {R_EARTH/1e6:.3f} × 10⁶ м")
print(f"    T0 = sqrt(R^3/GM) = {T0:.1f} с  (~{T0/60:.1f} мин)")
print(f"    V0 = sqrt(GM/R)   = {V0:.1f} м/с")
print()
for sat in SATELLITES:
    eps = sat['Cd'] * sat['S'] * RHO0 * R_EARTH / sat['m']
    beta = sat['m'] / (sat['Cd'] * sat['S'])
    print(f"    ε({sat['name']}) = Cd·S·ρ₀·R/m = {eps:.3e}   β = {beta:.1f} кг/м²")

# --- Основной расчёт ---
print()
results = []
for sat in SATELLITES:
    Cd, S, m = sat['Cd'], sat['S'], sat['m']
    beta = m / (Cd * S)

    # Аналитика
    T_analytic = analytic_time(400e3, 300e3, Cd, S, m)

    # Разностная схема по виткам
    t_num, h_num = orbit_iteration(Cd, S, m, h_start=400e3, h_stop=80e3, save_every=10)

    T_num_300 = time_at_altitude(t_num, h_num, 300e3)
    T_num_120 = time_at_altitude(t_num, h_num, 120e3)
    error = abs(T_analytic - T_num_300) / T_analytic * 100

    N_orb_300 = int(T_num_300 / (2 * np.pi * np.sqrt((R_EARTH + 350e3)**3 / GM)))
    N_orb_dec = int(T_num_120 / (2 * np.pi * np.sqrt((R_EARTH + 350e3)**3 / GM)))

    results.append({
        'sat': sat, 'beta': beta,
        'T_analytic': T_analytic,
        'T_num_300': T_num_300,
        'T_num_120': T_num_120,
        'error': error,
        't': t_num,
        'h': h_num / 1e3,   # в км
        'N_300': N_orb_300,
        'N_dec': N_orb_dec,
    })

    T_orb_400 = 2 * np.pi * np.sqrt((R_EARTH + 400e3)**3 / GM)
    da_400  = 2 * np.pi * (Cd*S/m) * atm_density(400e3) * (R_EARTH + 400e3)**2

    print(f"  [{sat['name']}] {sat['label']}")
    print(f"      β = {beta:.1f} кг/м²")
    print(f"      Орб. период на h=400 км : {T_orb_400/60:.2f} мин")
    print(f"      Δa за 1 виток (h=400 км): {da_400:.3f} м/виток")
    print(f"      Аналит. время 400→300 км: {T_analytic/86400:.1f} сут")
    print(f"      Числен. время 400→300 км: {T_num_300/86400:.1f} сут  "
          f"(погрешность {error:.2f}%,  ~{N_orb_300} витков)")
    print(f"      Время полн. схода с орб : {T_num_120/86400:.1f} сут  "
          f"= {T_num_120/3.156e7:.2f} лет  (~{N_orb_dec} витков)\n")

# 9. ПОСТРОЕНИЕ ГРАФИКОВ
plt.rcParams.update({
    'font.size': 9.5, 'axes.titlesize': 10.5,
    'axes.labelsize': 9.5, 'legend.fontsize': 8.5,
    'lines.linewidth': 2.0, 'figure.dpi': 130,
    'font.family': 'DejaVu Sans',
    'axes.grid': True, 'grid.alpha': 0.25,
})

fig = plt.figure(figsize=(18, 15))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.42)

# ─── (а) Высота орбиты от времени ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for r in results:
    ax1.plot(r['t'] / 86400, r['h'],
             color=r['sat']['color'], ls=r['sat']['ls'], label=r['sat']['label'])
ax1.axhline(400, color='#888', lw=1.0, ls='--', alpha=0.7, label='h = 400 км')
ax1.axhline(300, color='#888', lw=1.0, ls=':',  alpha=0.7, label='h = 300 км')
ax1.axhline(120, color='#8B2500', lw=1.2, ls='-.', alpha=0.9, label='h = 120 км (сход)')
ax1.fill_between(
    [0, max(r['t'][-1] for r in results) / 86400],
    80, 120, alpha=0.08, color='red', label='Плотные слои (вход)')
ax1.set_xlabel('Время, сут')
ax1.set_ylabel('Высота орбиты, км')
ax1.set_title('(а) Высота орбиты в зависимости от времени\n'
              '(разностная схема по виткам, шаг = 1 орбитальный период)')
ax1.legend(ncol=2, loc='upper right', fontsize=8.5)
ax1.set_xlim(left=0);  ax1.set_ylim(bottom=80)

# ─── (б) Орбитальная скорость от времени ────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
for r in results:
    h_m = r['h'] * 1e3
    v   = np.sqrt(GM / (R_EARTH + h_m)) / 1e3   # км/с
    ax2.plot(r['t'] / 86400, v,
             color=r['sat']['color'], ls=r['sat']['ls'], label=r['sat']['name'])
ax2.set_xlabel('Время, сут')
ax2.set_ylabel('Первая косм. скорость, км/с')
ax2.set_title('(б) Орбитальная скорость\nот времени')
ax2.legend(title='Спутник')
ax2.set_xlim(left=0)

# ─── (в) Профиль плотности атмосферы ──────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
h_range = np.linspace(100, 600, 500) * 1e3
ax3.semilogy(h_range / 1e3, atm_density(h_range), 'navy', lw=2.5,
             label='Экспон. модель')
ax3.semilogy(H_CHECK / 1e3, RHO_CHECK, 'ro', ms=7, zorder=5,
             label='Данные COESA-76')
ax3.axvspan(300, 400, alpha=0.13, color='orange', label='Диапазон 300–400 км')
ax3.axvline(300, color='orange', lw=1.2, ls='--')
ax3.axvline(400, color='orange', lw=1.2, ls='--')
ax3.set_xlabel('Высота h, км')
ax3.set_ylabel('Плотность ρ, кг/м³')
ax3.set_title('(в) Профиль плотности атмосферы')
ax3.legend(fontsize=8.5)
ax3.text(415, 8e-12,
         r'$\rho = \rho_0\,e^{-(h-300)/H}$' + '\n$H = 52$ км',
         fontsize=9.5, color='navy',
         bbox=dict(facecolor='lightyellow', alpha=0.9,
                   edgecolor='navy', boxstyle='round,pad=0.4'))

# ─── (г) delta_a за один виток в зависимости от высоты ──────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
h_plot = np.linspace(150, 500, 400)
for r in results:
    Cd_, S_, m_ = r['sat']['Cd'], r['sat']['S'], r['sat']['m']
    B_ = Cd_ * S_ / m_
    a_pl = R_EARTH + h_plot * 1e3
    da = 2 * np.pi * B_ * atm_density(h_plot * 1e3) * a_pl**2
    ax4.semilogy(h_plot, da, color=r['sat']['color'], ls=r['sat']['ls'],
                 label=r['sat']['label'])
ax4.axvline(300, color='gray', lw=0.9, ls=':')
ax4.axvline(400, color='gray', lw=0.9, ls='--')
ax4.set_xlabel('Высота h, км')
ax4.set_ylabel('|Δa| за 1 виток, м')
ax4.set_title('(г) Снижение орбиты за один виток\n'
              r'$|\Delta a| = \frac{2\pi C_d S}{m}\,\rho(a)\cdot a^2$')
ax4.legend(fontsize=8)

# ─── (д) Аналитика / Числ. схема по виткам ───────────────────────
ax5 = fig.add_subplot(gs[1, 2])
names_bar = [r['sat']['name'] for r in results]
T_anal_days = [r['T_analytic'] / 86400 for r in results]
T_num_days = [r['T_num_300']  / 86400 for r in results]
x_pos = np.arange(len(names_bar))
w = 0.32
bars_a = ax5.bar(x_pos - w/2, T_anal_days, w, label='Аналитич. формула',
                 color='steelblue', alpha=0.88)
bars_n = ax5.bar(x_pos + w/2, T_num_days,  w, label='Схема по виткам',
                 color='darkorange', alpha=0.88)
for bars in [bars_a, bars_n]:
    for b in bars:
        ax5.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                 f'{b.get_height():.0f}', ha='center', fontsize=8)
ax5.set_xticks(x_pos);  ax5.set_xticklabels(names_bar)
ax5.set_ylabel('Время спуска 400→300 км, сут')
ax5.set_title('(д) Сравнение: аналитика /\nразностная схема по виткам')
ax5.legend(fontsize=8.5)

# ─── (е) 2D Траектория (схема Эйлера в декартовых координатах) ────
ax6 = fig.add_subplot(gs[2, 0], aspect='equal')
sat_2d = SATELLITES[1]   # Спутник Б (наибольшее торможение)
x_2d, y_2d = simulate_2d(sat_2d['Cd'], sat_2d['S'], sat_2d['m'],
                          h_start=400e3, n_orbits=4, tau=10.0)
x_km = x_2d / 1e3;  y_km = y_2d / 1e3
pts = np.array([x_km, y_km]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segs, cmap='plasma',
                       norm=plt.Normalize(0, len(segs)),
                       linewidths=0.9, alpha=0.85)
lc.set_array(np.arange(len(segs)))
ax6.add_collection(lc)
ax6.add_patch(Circle((0, 0), R_EARTH / 1e3, color='deepskyblue',
                      zorder=5, label='Земля'))
ax6.plot(x_km[0], y_km[0], 'go', ms=9, zorder=10, label='Начало (h=400 км)')
lim = (R_EARTH + 435e3) / 1e3
ax6.set_xlim(-1.5*lim, 1.5*lim);  ax6.set_ylim(-1.5*lim, 1.5*lim)
ax6.set_xlabel('x, км');   ax6.set_ylabel('y, км')
ax6.set_title(f'(е) Двумерная траектория — {sat_2d["label"]}\n'
              '(4 витка, схема Эйлера 2D, τ = 10 с)')
ax6.legend(fontsize=8)
cbar = plt.colorbar(lc, ax=ax6, fraction=0.03, pad=0.04)
cbar.set_label('Прогресс по времени', fontsize=7.5)

# ─── (ж) Итоговые времена по трём спутникам ───────────────────────
ax7 = fig.add_subplot(gs[2, 1:])
x3 = np.arange(len(results))
T_300_days = [r['T_num_300'] / 86400 for r in results]
T_120_days = [r['T_num_120'] / 86400 for r in results]
bar_colors = [r['sat']['color'] for r in results]
bar_labels = [r['sat']['label'] for r in results]

b1 = ax7.bar(x3 - 0.22, T_300_days, 0.40,
             label='Спуск 400 → 300 км', color=bar_colors,
             alpha=0.55, edgecolor='k', linewidth=0.7)
b2 = ax7.bar(x3 + 0.22, T_120_days, 0.40,
             label='Полный сход с орбиты (→ 120 км)', color=bar_colors,
             alpha=1.00, edgecolor='k', linewidth=0.7)
for b in b1:
    ax7.text(b.get_x() + b.get_width()/2, b.get_height() + 2,
             f'{b.get_height():.0f} сут', ha='center', fontsize=8.5)
for b in b2:
    ax7.text(b.get_x() + b.get_width()/2, b.get_height() + 2,
             f'{b.get_height():.0f} сут', ha='center', fontsize=8.5)
ax7.set_xticks(x3);  ax7.set_xticklabels(bar_labels, fontsize=9)
ax7.set_ylabel('Время, сут')
ax7.set_title('(ж) Время спуска и полного схода с орбиты\n'
              '(разностная схема по виткам,  начало h₀ = 400 км)')
ax7.legend(fontsize=9)

# --- Общий заголовок ---
fig.suptitle(
    'Движение спутника в атмосфере Земли при экспоненциальном распределении плотности воздуха\n'
    r'$\rho(h) = \rho_0\,e^{-(h-300)/H}$,  $H = 52$ км,  '
    r'$\rho_0 = 1{,}916\!\times\!10^{-11}$ кг/м³   |   '
    r'Модель: $\Delta a_k = -\frac{2\pi C_d S}{m}\,\rho(a_k)\,a_k^2$',
    fontsize=11.5, fontweight='bold', y=0.998)

plt.savefig('satellite_main.png', bbox_inches='tight', dpi=130)
plt.close()
print("\nОсновной график сохранён: satellite_main.png")

# 10. ДОПОЛНИТЕЛЬНЫЙ ГРАФИК — сходимость схемы по виткам
fig2, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig2.suptitle(
    'Сходимость разностной схемы по виткам — Спутник А\n'
    'Сравнение: группировка по 1, 5, 20 виткам за шаг',
    fontsize=11, fontweight='bold')

r0 = results[0]
s0 = r0['sat']
Cd0, S0, m0 = s0['Cd'], s0['S'], s0['m']

for step_group, color, label in [(1,  '#1f78b4', '1 виток/шаг (точный)'),
                                  (5,  '#e31a1c', '5 витков/шаг'),
                                  (20, '#33a02c', '20 витков/шаг')]:
    B_ = Cd0 * S0 / m0
    a_ = R_EARTH + 400e3
    t_ = 0.0
    ts_, hs_ = [t_], [a_ - R_EARTH]
    while (a_ - R_EARTH) > 80e3:
        for _ in range(step_group):
            rho_ = atm_density(a_ - R_EARTH)
            a_  += -2 * np.pi * B_ * rho_ * a_**2
            t_  += 2 * np.pi * np.sqrt(a_**3 / GM)
            if (a_ - R_EARTH) <= 80e3:
                break
        ts_.append(t_);  hs_.append(a_ - R_EARTH)
    ts_ = np.array(ts_);  hs_ = np.array(hs_) / 1e3
    ls_ = '-' if step_group == 1 else ('--' if step_group == 5 else ':')
    axes[0].plot(ts_ / 86400, hs_, color=color, ls=ls_, lw=1.8, label=label)

    if step_group != 1:
        t_base, h_base = r0['t'], r0['h']
        t_common  = np.linspace(0, min(ts_[-1], t_base[-1]), 1500)
        h_interp1 = np.interp(t_common, ts_,   hs_)
        h_interp2 = np.interp(t_common, t_base, h_base)
        diff = h_interp1 - h_interp2
        axes[1].plot(t_common / 86400, diff, color=color, ls=ls_, lw=1.6,
                     label=f'{step_group} витков/шаг')

for ax_i, (title, xl, yl) in zip(axes, [
    ('Высота орбиты: разные шаги группировки', 'Время, сут', 'Высота, км'),
    ('Ошибка схемы по сравнению с шагом 1 виток', 'Время, сут', 'Δh, км')]):
    ax_i.set_title(title);  ax_i.set_xlabel(xl);  ax_i.set_ylabel(yl)
    ax_i.legend();  ax_i.grid(alpha=0.3)

plt.savefig('satellite_convergence.png', bbox_inches='tight', dpi=130)
plt.close()
print("График сходимости сохранён: satellite_convergence.png")

# 11. ИТОГОВАЯ ТАБЛИЦА И ВЫВОДЫ
print("\n" + "=" * 65)
print("  ИТОГОВЫЕ РЕЗУЛЬТАТЫ ВЫЧИСЛИТЕЛЬНОГО ЭКСПЕРИМЕНТА")
print("=" * 65)
print(f"  {'Спутник':<28} {'β, кг/м²':>10} {'T(400→300), сут':>16} {'T_сход, сут':>12}")
print("  " + "─" * 66)
for r in results:
    print(f"  {r['sat']['label']:<28} {r['beta']:>10.1f} "
          f"{r['T_num_300']/86400:>16.1f} {r['T_num_120']/86400:>12.1f}")

print()
print("  Скорость снижения Δa за виток (Спутник А):")
s_a = results[0]['sat']
for h_km in [400, 350, 300, 250, 200]:
    a_ = R_EARTH + h_km * 1e3
    B_ = s_a['Cd'] * s_a['S'] / s_a['m']
    da = 2 * np.pi * B_ * atm_density(h_km * 1e3) * a_**2
    T_ = 2 * np.pi * np.sqrt(a_**3 / GM)
    print(f"    h = {h_km} км: |Δa| = {da:.3f} м/виток  =  {da/T_*86400:.3f} км/сут")

