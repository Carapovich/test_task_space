import sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.integrate as integrator

import src.core as core
import src.utils as utils

# Частота печати в файл в Гц
PRINT_FREQUENCY = 100
# Частота кадров анимации в Гц
ANIM_FREQUENCY = 20


class SimulationInput:
    def __init__(self,
                 a, e, periapsis, long, i, m0, spr_k, spr_l0, spr_l1, spr_l2, yaw, pitch, mass_lv, mass_sc, sim_time):
        self.semi_major = a
        self.eccentricity = e
        self.arg_periapsis = periapsis
        self.long_ascend = long
        self.inclination = i
        self.mean_anomaly = m0
        self.spring_stiffness = spr_k
        self.spring_l0 = spr_l0
        self.spring_l1 = spr_l1
        self.spring_l2 = spr_l2
        self.start_yaw = yaw
        self.start_pitch = pitch
        self.mass_lv = mass_lv
        self.mass_sc = mass_sc
        self.sim_time = sim_time

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            a=d['semi major'] * 1e3,
            e=d['eccentricity'],
            periapsis=np.radians(d['arg periapsis']),
            long=np.radians(d['long ascend']),
            i=np.radians(d['inclination']),
            m0=np.radians(d['mean anomaly']),
            spr_k=d['spring k'],
            spr_l0=d['spring l0'] * 1e-3,
            spr_l1=d['spring l1'] * 1e-3,
            spr_l2=d['spring l2'] * 1e-3,
            yaw=np.radians(d['start yaw']),
            pitch=np.radians(d['start pitch']),
            mass_lv=d['mass lv'],
            mass_sc=d['mass sc'],
            sim_time=d['sim time']
        )


def run_simulation(sim_input: SimulationInput) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # Задание положения РН и КА
    state_lv = core.kepler2eci(sim_input.semi_major, sim_input.eccentricity, sim_input.inclination,
                               sim_input.long_ascend, sim_input.arg_periapsis, sim_input.mean_anomaly)
    e_lv2sc, state_sc = core.get_state_relative(state_lv, sim_input.spring_l2,
                                                sim_input.start_yaw, sim_input.start_pitch)

    # Настройка параметров и расчет решения системы ДУ движения
    spring: dict = {"k": sim_input.spring_stiffness, "l0": sim_input.spring_l0, "l1": sim_input.spring_l1}
    t_eval = np.linspace(start=0., stop=sim_input.sim_time, num=int(PRINT_FREQUENCY * sim_input.sim_time + 1))
    result = integrator.solve_ivp(core.motion_equation_rhs,
                                  t_span=(t_eval[0], t_eval[-1]),
                                  y0=np.vstack((state_lv, state_sc)).ravel(),
                                  t_eval=t_eval,
                                  rtol=1e2 * sys.float_info.epsilon,
                                  events=core.event_decoupling,
                                  args=(spring, sim_input.mass_lv, sim_input.mass_sc, e_lv2sc))

    # Расчет динамики силы пружинного толкателя на печать
    rel_r = la.norm(result.y[:3, :] - result.y[6:9, :], axis=0)
    dx = np.where(rel_r >= spring['l1'], spring['l0'], rel_r)
    stiff_force = spring["k"] * (spring["l0"] - dx)

    return t_eval, result.y, stiff_force, result.t_events[0][0]


def process_result(sim_result: tuple):
    f_out, t, y, stiff_force, t_decoupling = sim_result
    lv_r, lv_v, sc_r, sc_v = np.vsplit(y, 4)
    rel_r = la.norm(lv_r - sc_r, axis=0)
    rel_v = la.norm(lv_v - sc_v, axis=0)

    # Печать в csv-файл
    res_keys = [
        'time, с',
        'lv Rx, м', 'lv Ry, м', 'lv Rz, м', 'lv Vx, м/с', 'lv Vy, м/с', 'lv Vz, м/с',
        'sc Rx, м', 'sc Ry, м', 'sc Rz, м', 'sc Vx, м/с', 'sc Vy, м/с', 'sc Vz, м/с',
        'rel dist, м', 'rel vel, м/с', 'stiff force, Н'
    ]
    res_values = list(np.vstack((t, y, rel_r, rel_v, stiff_force)).T)
    utils.print_results(f_out, [dict(zip(res_keys, col)) for col in res_values])

    # Линейная экстраполяция для получения референсных прямых для проекций вектора скорости на увеличенном графике
    i0 = np.argwhere(t > 2. * t_decoupling).ravel()[0]
    ref_lv_v = np.stack((lv_v[:, i0] - (lv_v[:, 2 * i0] - lv_v[:, i0]) / (t[2 * i0] - t[i0]) * t[i0], lv_v[:, i0])).T
    ref_sc_v = np.stack((sc_v[:, i0] - (sc_v[:, 2 * i0] - sc_v[:, i0]) / (t[2 * i0] - t[i0]) * t[i0], sc_v[:, i0])).T
    ref_t = t[[0, i0]]

    # Печати графиков
    utils.plot_vector(t, [lv_r, lv_v],
                      'Проекции радиус-вектора и вектора скорости РН в ECI J2000',
                      [r'Проекция радиус-вектора $\bf r\it_x$', r'Проекция радиус-вектора $\bf r\it_y$',
                       r'Проекция радиус-вектора $\bf r\it_z$',
                       r'Проекция вектора скорости $\bf v\it_x$', r'Проекция вектора скорости $\bf v\it_y$',
                       r'Проекция вектора скорости $\bf v\it_z$'],
                      [r'$\bf r\it_x^I$, метры', r'$\bf r\it_y^I$, метры', r'$\bf r\it_z^I$, метры',
                       r'$\bf v\it_x^I$, м/сек', r'$\bf v\it_y^I$, м/сек', r'$\bf v\it_z^I$, м/сек'])

    utils.plot_vector(t, [sc_r, sc_v],
                      'Проекции радиус-вектора и вектора скорости КА в ECI J2000',
                      [r'Проекция радиус-вектора $\bf r\it_x$', r'Проекция радиус-вектора $\bf r\it_y$',
                       r'Проекция радиус-вектора $\bf r\it_z$',
                       r'Проекция вектора скорости $\bf v\it_x$', r'Проекция вектора скорости $\bf v\it_y$',
                       r'Проекция вектора скорости $\bf v\it_z$'],
                      [r'$\bf r\it_x^I$, метры', r'$\bf r\it_y^I$, метры', r'$\bf r\it_z^I$, метры',
                       r'$\bf v\it_x^I$, м/сек', r'$\bf v\it_y^I$, м/сек', r'$\bf v\it_z^I$, м/сек'])

    utils.plot_vector_with_ref_line(t[:i0 + 1], [lv_v[:, :i0 + 1], sc_v[:, :i0 + 1]], ref_t, [ref_lv_v, ref_sc_v],
                                    r'Проекции вектора скорости РН и КА в ECI J2000 на начальном этапе моделирования',
                                    [r'Проекция $\bf v\it_x$ РН', r'Проекция $\bf v\it_y$ РН',
                                     r'Проекция $\bf v\it_z$ РН',
                                     r'Проекция $\bf v\it_x$ КА', r'Проекция $\bf v\it_y$ КА',
                                     r'Проекция $\bf v\it_z$ КА'],
                                    [r'$\bf v\it_x^I$, м/сек', r'$\bf v\it_y^I$, м/сек', r'$\bf v\it_z^I$, м/сек',
                                     r'$\bf v\it_x^I$, м/сек', r'$\bf v\it_y^I$, м/сек', r'$\bf v\it_z^I$, м/сек'])

    utils.plot_vector(t, np.vstack((rel_r, rel_v, stiff_force)),
                      'Относительное расстояние, скорость и сила пружинного толкателя',
                      ['Отн. расстояние между РН и КА', 'Отн. скорость между РН и КА', 'Сила пружинного толкателя'],
                      [r'$\bf |r|\it_{отн}$, метры', r'$\bf |v|\it_{отн}$, м/сек', r'$\bf F\it_{упр}$, Н'],
                      subplot_order=(3, 1), single_scale_y=False)

    fig = plt.figure(num='Траектории РН и КА в пространстве')
    utils.show_anim(fig, utils.plot_vehicles_trajectory, ANIM_FREQUENCY, t, y)
