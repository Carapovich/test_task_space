"""
simulation.py

Контейнер входных данных, алгоритмы расчета программы
моделирования и печати его результов в разных форматах
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.integrate as integrator

import src.core as core
import src.utils as utils


class SimulationInput:
    """
    Класс-контейнер (структура) для хранения входных данных моделирования

    Экземпляр класса должен создаваться через метод ``from_dict``, принимающий
    соответствующий словарь, полученный из входного csv-файла

    Поля класса
    ----------------
    - ``semi_major``        --> Большая полуось начальной орбиты (метры)
    - ``eccentricity``      --> Эксцентриситет начальной орбиты (б/р)
    - ``arg_periapsis``     --> Аргумент перицентра начальной орбиты (радианы)
    - ``long_ascend``       --> Долгота восходящего узла начальной орбиты (радианы)
    - ``inclination``       --> Угол наклонения начальной орбиты (радианы)
    - ``mean_anomaly``      --> Средняя аномалия начальной орбиты (радианы)
    - ``spring_stiffness``  --> Коэффициент жесткости пружины толкателя (Н/м)
    - ``spring_l0``         --> Длина пружины толкателя в недеформированном состоянии (метры)
    - ``spring_l1``         --> Конечная длина пружины толкателя (упор) (метры)
    - ``spring_l2``         --> Начальная длина пружины толкателя (метры)
    - ``start_yaw``         --> Угол "рысканья" направления действия силы
                                толкателя, относительно начального вектора скорости РН (радианы)
    - ``start_pitch``       --> Угол "тангажа" направления действия силы
                                толкателя, относительно начального вектора скорости РН (радианы)
    - ``mass_lv``           --> Масса РН (кг)
    - ``mass_lv``           --> Масса КА (кг)
    - ``sim_time``          --> Продолжительность моделирования (секунды)
    """

    def __init__(self, a, e, periapsis, long, i, m0, spr_k, spr_l0, spr_l1, spr_l2,
                 yaw, pitch, mass_lv, mass_sc, sim_time, print_freq, anim_freq):
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
        self.print_frequency = print_freq
        self.anim_frequency = anim_freq

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
            sim_time=d['sim time'],
            print_freq=d['print freq'],
            anim_freq=d['anim freq']
        )


class SimulationOutput:
    """
    Класс-контейнер (структура) для хранения результатов моделирования
    """

    def __init__(self, t, lv_r, lv_v, sc_r, sc_v, rel_r, rel_v, f_s, t_decoupling, anim_freq):
        self.timestamps = t
        """ Массив моментов времени, для которых рассчитано решение задачи моделирования (сек.) """
        self.lv_r = lv_r
        """ Массив решений радиус-векторов РН от времени (метры) """
        self.lv_v = lv_v
        """ Массив решений векторов скорости РН от времени (м/сек) """
        self.sc_r = sc_r
        """ Массив решений радиус-векторов КА от времени (метры) """
        self.sc_v = sc_v
        """ Массив решений векторов скорости КА от времени (м/сек) """
        self.rel_r = rel_r
        """ Массив решений относительных расстояний между РН и КА (метры) """
        self.rel_v = rel_v
        """ Массив решений относительных скоростей между РН и КА (метры) """
        self.stiff_forces = f_s
        """ Массив решений сил пружинного толкателя от времени (Н) """
        self.time_decoupling = t_decoupling
        """ Момент времени, в который произошло отделение КА от РН (сек.) """
        self.anim_frequency = anim_freq
        """ Частота печати результатов моделирования в анимации """

    def __iter__(self):
        return iter(np.vstack((
            self.timestamps,
            self.lv_r, self.lv_v,
            self.sc_r, self.sc_v,
            self.rel_r, self.rel_v,
            self.stiff_forces
        )).T)


def run_simulation(sim_input: SimulationInput) -> SimulationOutput:
    """
    Рассчитывает задачу моделирования динамики отделения КА от ступени РН
    по входным данным и возвращает требуемые результаты моделирования

    ----------------

    На первом этапе рассчитывает положение и вектор скорости РН и КА по
    заданным кеплеровым элементам орбиты и относительному положению и
    ориентации аппаратов друг относительно друга. \n
    На втором этапе рассчитывает решение системы ОДУ динамики и кинематики
    поступательного движения аппаратов методом Рунге-Кутты 4-го порядка в
    указанные моменты времени по начальным условиям, рассчитанным на первом этапе. \n
    На третьем этапе рассчитываются:
        - временной закон силы пружинного толкателя,
        - относительное расстояние между РН и КА,
        - относительная скорость между РН и КА,
    по полученным из решения ОДУ радиус-векторам и векторам скорости аппаратов.

    ----------------

    :param sim_input: Входные данные задачи моделирования в виде класса-контейнера
    :return: Результаты решения задачи моделирования в виде класса-контейнера
    """
    # Задание начальных координат и скоростей РН и КА
    state_lv = core.kepler2eci(sim_input.semi_major, sim_input.eccentricity, sim_input.inclination,
                               sim_input.long_ascend, sim_input.arg_periapsis, sim_input.mean_anomaly)
    e_lv2sc, state_sc = core.get_state_relative(state_lv, sim_input.spring_l2,
                                                sim_input.start_yaw, sim_input.start_pitch)

    # Настройка параметров и расчет решения системы ДУ движения
    spring: dict = {"k": sim_input.spring_stiffness, "l0": sim_input.spring_l0, "l1": sim_input.spring_l1}
    t_eval = np.linspace(start=0., stop=sim_input.sim_time, num=int(sim_input.print_frequency * sim_input.sim_time + 1))
    result = integrator.solve_ivp(core.motion_equation_rhs,
                                  t_span=(t_eval[0], t_eval[-1]),
                                  y0=np.vstack((state_lv, state_sc)).ravel(),
                                  t_eval=t_eval,
                                  rtol=1e2 * sys.float_info.epsilon,
                                  events=core.event_decoupling,
                                  args=(spring, sim_input.mass_lv, sim_input.mass_sc, e_lv2sc))

    # Расчет относительного расстояния и скорости между РН и КА от времени
    lv_r, lv_v, sc_r, sc_v = np.vsplit(result.y, 4)
    rel_r = la.norm(lv_r - sc_r, axis=0)
    rel_v = la.norm(lv_v - sc_v, axis=0)

    # Расчет динамики силы пружинного толкателя  от времени
    dx = np.where(rel_r >= spring['l1'], spring['l0'], rel_r)
    stiff_force = spring["k"] * (spring["l0"] - dx)

    return SimulationOutput(
        t=t_eval,
        lv_r=lv_r,
        lv_v=lv_v,
        sc_r=sc_r,
        sc_v=sc_v,
        rel_r=rel_r,
        rel_v=rel_v,
        f_s=stiff_force,
        t_decoupling=result.t_events[0][0],
        anim_freq=sim_input.anim_frequency
    )


def process_result(sim_out: SimulationOutput, filename_csv: str):
    """
    Печает результаты моделирования динамики отделения КА от ступени РН в виде:
        - csv-файла,
        - плоских графиков проекций радиус-векторов и векторов РН и КА от времени
        - плоских графиков отн. расстояния, отн. скорости между РН и КА и силы пружинного толкателя от времени
        - пространственного графика траектории РН и КА (с анимацией движения тел)

    ----------------

    :param sim_out: Результаты решения задачи моделирования в виде класса-контейнера
    :param filename_csv: Путь к csv-файлу, в который печатаются результаты моделирования
    """
    # Печать в csv-файл
    res_keys = [
        'time, с',
        'lv Rx, м', 'lv Ry, м', 'lv Rz, м', 'lv Vx, м/с', 'lv Vy, м/с', 'lv Vz, м/с',
        'sc Rx, м', 'sc Ry, м', 'sc Rz, м', 'sc Vx, м/с', 'sc Vy, м/с', 'sc Vz, м/с',
        'rel dist, м', 'rel vel, м/с', 'stiff force, Н'
    ]
    utils.print_results(filename_csv, [dict(zip(res_keys, col)) for col in sim_out])

    # Линейная экстраполяция для получения опорных прямых для проекций вектора скорости на увеличенном графике
    i0 = np.argwhere(sim_out.timestamps > 2. * sim_out.time_decoupling).ravel()[0]
    ref_lv_v = np.stack((sim_out.lv_v[:, i0] - (sim_out.lv_v[:, 2 * i0] - sim_out.lv_v[:, i0]) /
                         (sim_out.timestamps[2 * i0] - sim_out.timestamps[i0]) * sim_out.timestamps[i0],
                         sim_out.lv_v[:, i0])).T
    ref_sc_v = np.stack((sim_out.sc_v[:, i0] - (sim_out.sc_v[:, 2 * i0] - sim_out.sc_v[:, i0]) /
                         (sim_out.timestamps[2 * i0] - sim_out.timestamps[i0]) * sim_out.timestamps[i0],
                         sim_out.sc_v[:, i0])).T
    ref_t = sim_out.timestamps[[0, i0]]

    # Печати графиков
    utils.plot_vector(sim_out.timestamps, [sim_out.lv_r, sim_out.lv_v],
                      'Проекции радиус-вектора и вектора скорости РН в ECI J2000',
                      [r'Проекция радиус-вектора $\bf r\it_x$', r'Проекция радиус-вектора $\bf r\it_y$',
                       r'Проекция радиус-вектора $\bf r\it_z$',
                       r'Проекция вектора скорости $\bf v\it_x$', r'Проекция вектора скорости $\bf v\it_y$',
                       r'Проекция вектора скорости $\bf v\it_z$'],
                      [r'$\bf r\it_x^I$, метры', r'$\bf r\it_y^I$, метры', r'$\bf r\it_z^I$, метры',
                       r'$\bf v\it_x^I$, м/сек', r'$\bf v\it_y^I$, м/сек', r'$\bf v\it_z^I$, м/сек'])

    utils.plot_vector(sim_out.timestamps, [sim_out.sc_r, sim_out.sc_v],
                      'Проекции радиус-вектора и вектора скорости КА в ECI J2000',
                      [r'Проекция радиус-вектора $\bf r\it_x$', r'Проекция радиус-вектора $\bf r\it_y$',
                       r'Проекция радиус-вектора $\bf r\it_z$',
                       r'Проекция вектора скорости $\bf v\it_x$', r'Проекция вектора скорости $\bf v\it_y$',
                       r'Проекция вектора скорости $\bf v\it_z$'],
                      [r'$\bf r\it_x^I$, метры', r'$\bf r\it_y^I$, метры', r'$\bf r\it_z^I$, метры',
                       r'$\bf v\it_x^I$, м/сек', r'$\bf v\it_y^I$, м/сек', r'$\bf v\it_z^I$, м/сек'])

    utils.plot_vector_with_ref_line(sim_out.timestamps[:i0 + 1], [sim_out.lv_v[:, :i0 + 1], sim_out.sc_v[:, :i0 + 1]],
                                    ref_t, [ref_lv_v, ref_sc_v],
                                    r'Проекции вектора скорости РН и КА в ECI J2000 на начальном этапе моделирования',
                                    [r'Проекция $\bf v\it_x$ РН', r'Проекция $\bf v\it_y$ РН',
                                     r'Проекция $\bf v\it_z$ РН',
                                     r'Проекция $\bf v\it_x$ КА', r'Проекция $\bf v\it_y$ КА',
                                     r'Проекция $\bf v\it_z$ КА'],
                                    [r'$\bf v\it_x^I$, м/сек', r'$\bf v\it_y^I$, м/сек', r'$\bf v\it_z^I$, м/сек',
                                     r'$\bf v\it_x^I$, м/сек', r'$\bf v\it_y^I$, м/сек', r'$\bf v\it_z^I$, м/сек'])

    utils.plot_vector(sim_out.timestamps, np.vstack((sim_out.rel_r, sim_out.rel_v, sim_out.stiff_forces)),
                      'Относительное расстояние, скорость и сила пружинного толкателя',
                      ['Отн. расстояние между РН и КА', 'Отн. скорость между РН и КА', 'Сила пружинного толкателя'],
                      [r'$\bf |r|\it_{отн}$, метры', r'$\bf |v|\it_{отн}$, м/сек', r'$\bf F\it_{упр}$, Н'],
                      subplot_order=(3, 1), single_scale_y=False)

    fig = plt.figure(num='Траектории РН и КА в пространстве')
    results_states = np.vstack((sim_out.lv_r, sim_out.lv_v, sim_out.sc_r, sim_out.sc_v))
    utils.show_anim(fig, utils.plot_vehicles_trajectory, sim_out.anim_frequency, sim_out.timestamps, results_states)
