"""
core.py

Реализация математического аппарата
программы моделирования
"""

import sys
import numpy as np
import numpy.linalg as la

from scipy.spatial.transform import Rotation as R

# Значение гравитационного параметра для Земли
EARTH_FM = 398600.4418e+9


def kepler2eci(a: float, e: float, i: float, long: float, periapsis: float, m0: float) -> np.ndarray:
    """
    Рассчитывает радиус-вектор и вектор скорости материальной точки на эллиптической орбите
    в проекциях на ИГЦСК по заданным кеплеровым элементам орбиты

    ----------------

    :param a: Большая полуось в метрах
    :param e: Эксцентриситет (0 < ``e`` < 1)
    :param i: Наклонение в радианах
    :param long: Долгота восходящего узла в радианах
    :param periapsis: Аргумент перицентра в радианах
    :param m0: Средняя аномалия в радианах

    :return: Матрица 2х3, в первой строке проекции радиус-вектора, во второй – вектора скорости материальной точки

    Ссылки на литературу
    ----------------
        M.Eng. René Schwarz, "Keplerian Orbit Elements → Cartesian State Vectors (Memorandum № 1)",
        https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf

    """

    # Расчет эксцентрической аномалии
    anom_e, d_anom_e = m0, 1.
    while abs(d_anom_e) >= 2 * sys.float_info.epsilon:
        d_anom_e = (anom_e - e * np.sin(anom_e) - m0) / (1 - e * np.cos(anom_e))
        anom_e -= d_anom_e

    # Истинной аномалии
    anom_nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(anom_e / 2.), np.sqrt(1 - e) * np.cos(anom_e / 2.))
    # Орбитального расстояния
    r_c = a * (1 - e * np.cos(anom_e))

    # Радиус-вектор и вектор скорости тела на орбите в ОСК
    r_o = r_c * np.array([np.cos(anom_nu), np.sin(anom_nu), 0.])
    v_o = np.sqrt(EARTH_FM * a) / r_c * np.array([-np.sin(anom_e), np.sqrt(1 - e ** 2) * np.cos(anom_e), 0.])

    # Радиус-вектор и вектор линейной скорости тела в ИГЦСК
    rot_o2i = R.from_euler('zxz', [periapsis, i, long])
    r_i, v_i = rot_o2i.apply([r_o, v_o])

    return np.vstack((r_i, v_i))


def get_state_relative(src_state: np.ndarray, distance: float, yaw: float, pitch: float) -> tuple:
    """
    Рассчитывает радиус-вектор точки, относительно заданной по известному расстоянию
    между ними и ориентации искомой точки

    ----------------

    Искомый радиус-вектор получается путем сложения заданного радиус-вектора и вектора-слагаемого.
    Длина вектора-слагаемого равна заданному расстоянию ``distance``. Ориентация вектора-слагаемого
    задается двумя последовательными поворотами вектора скорости исходной точки. Первый поворот на
    угол ``yaw`` осуществляется в плоскости, образованной радиус-вектором и вектором скорости исходной
    точки. Для второго поворота рассчитывается вектор-нормаль, как векторное произведение вектора
    скорости на радиус-вектор. Второй поворот на угол ``pitch`` осуществляется в плоскости повернутого
    на угол ``yaw`` вектора скорости исходной точки и рассчитанного вектора-нормали.

    ----------------

    :param src_state: Матрица 2х3, в первой строке проекции радиус-вектора, во второй – вектора скорости исходной точки
    :param distance: Расстояние между точками в метрах
    :param yaw: Перый угол поворота вектора скорости исходной точки в радианах
    :param pitch: Второй угол поворота вектора скорости исходной точки в радианах
    :return:
    Кортеж, первый элемент которого единичный вектор-слагаемое, а второй – матрица 2х3,
    в первой строке проекции радиус-вектора, во второй – вектора скорости искомой точки
    """

    vec_r, vec_v = src_state

    # Формирование базиса для поворота
    e_x, e_y = vec_v / la.norm(vec_v), -vec_r / la.norm(vec_r)
    e_z = la.cross(e_x, e_y)

    # Поворот вектора скорости
    rot_yaw = R.from_rotvec(yaw * e_z)
    rot_pitch = R.from_rotvec(-pitch * rot_yaw.apply(e_y))
    e_result = (rot_pitch * rot_yaw).apply(e_x)

    return e_result, np.vstack((vec_r + e_result * distance, vec_v))


def event_decoupling(t: float, y_vecs: np.ndarray, *constants) -> float:
    """
    Функция обработки события "расстыковки" между РН и КА при численном
    решении системы ДУ динамики по известной конечной длине пружины

    ----------------

    :param t: Не используется
    :param y_vecs: Решение системы ДУ в момент времени ``t``
    :param constants: Постоянные величины, необходимые для решения ДУ
                      (используется только конечная длина пружины)
    :return: Результат сравнения расстояния между РН и КА и конечной длиной пружины
    """

    lv_r, _, sc_r, _ = np.split(y_vecs, 4)
    spring = constants[0]

    return la.norm(lv_r - sc_r) < spring["l1"]


def motion_equation_rhs(t: float, y_vecs: np.ndarray, *constants) -> np.ndarray:
    """
    Функция расчета производной правой части системы ОДУ динамики поступательного движения РН и КА

    ----------------

    :param t: Момент времени решения в секундах (не используется)
    :param y_vecs: Решение системы ОДУ в момент времени ``t``
    :param constants: Постоянные величины, необходимые для решения ДУ: параметры пружины
                      (жесткость, конечная длина, длина в недеформированном состоянии),
                      масса РН и КА, единичный вектор ``e_lv2sc``, задающий направление
                      действия силы упругости пружинного толкателя
    :return: Значение производной правой части системы ОДУ
    """

    if not hasattr(motion_equation_rhs, 'decoupling'):
        motion_equation_rhs.decoupling = False

    lv_r, lv_v, sc_r, sc_v = np.split(y_vecs, 4)
    spring, lv_m, sc_m, e_lv2sc = constants

    # Расчет длины пружины
    delta_x = la.norm(lv_r - sc_r)
    if not motion_equation_rhs.decoupling and delta_x >= 2 * spring["l1"]:
        motion_equation_rhs.decoupling = True

    # Ускорения от пружины
    lv_s_a, sc_s_a = np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    if not motion_equation_rhs.decoupling and delta_x < spring["l1"]:
        stiff_force = spring["k"] * (spring["l0"] - delta_x)
        lv_s_a = -e_lv2sc * stiff_force / lv_m
        sc_s_a = +e_lv2sc * stiff_force / sc_m

    # Гравитационное ускорение
    lv_gm = -EARTH_FM * lv_r / la.norm(lv_r) ** 3
    sc_gm = -EARTH_FM * sc_r / la.norm(sc_r) ** 3

    dy_dt = np.hstack((lv_v, lv_gm + lv_s_a, sc_v, sc_gm + sc_s_a))

    return dy_dt
