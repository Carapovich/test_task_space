
import numpy as np
import numpy.linalg as la
import sys

from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

EARTH_FM = 398600.4418e+9




def kepler2eci(a, e, i, long, periapsis, m0) -> NDArray:
    # Расчет эксцентрической аномалии
    anom_e, d_anom_e = m0, 1.
    while np.abs(d_anom_e) >= 2 * sys.float_info.epsilon:
        d_anom_e = (anom_e - e * np.sin(anom_e) - m0) / (1 - e * np.cos(anom_e))
        anom_e -= d_anom_e

    # Расчет истинной аномалии
    anom_nu = 2 * np.arctan2( np.sqrt(1 + e) * np.sin(anom_e / 2.), np.sqrt(1 - e) * np.cos(anom_e / 2.) )
    # Расчет орбитального расстояния
    r_c = a * (1 - e * np.cos(anom_e))

    # Радиус-вектор и вектор скорости тела на орбите в ОСК
    r_o = r_c * np.array([ np.cos(anom_nu), np.sin(anom_nu), 0. ])
    v_o = np.sqrt(EARTH_FM * a) / r_c * np.array([ -np.sin(anom_e), np.sqrt(1 - e**2) * np.cos(anom_e), 0. ])

    # Радиус-вектор и вектор линейной скорости тела в ИГЦСК
    rot_o2i = R.from_euler('zxz', [periapsis, i, long])
    r_i, v_i = rot_o2i.apply([r_o, v_o])

    return np.vstack( (r_i, v_i) )

def get_state_relative(src_state, distance, yaw, pitch):
    vec_r, vec_v = src_state

    e_x, e_y = vec_v / la.norm(vec_v), -vec_r / la.norm(vec_r)
    e_z = la.cross(e_x, e_y)

    rot_yaw = R.from_rotvec(-yaw * e_z)
    rot_pitch = R.from_rotvec(pitch * rot_yaw.apply(e_x))
    e_result = (rot_pitch * rot_yaw).apply(e_y)

    return e_result, np.vstack( (vec_r + e_result * distance, vec_v) )

def motion_equation_rhs(t, y_vecs: np.ndarray, *constants) -> np.ndarray:
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
        lv_s_a = -e_lv2sc * spring["k"] / lv_m * (spring["l0"] - delta_x)
        sc_s_a = +e_lv2sc * spring["k"] / sc_m * (spring["l0"] - delta_x)

    # Гравитационное ускорение
    lv_gm = -EARTH_FM * lv_r / la.norm(lv_r)**3
    sc_gm = -EARTH_FM * sc_r / la.norm(sc_r)**3

    # Составление правых частей
    dy_dt = np.hstack((lv_v, lv_gm + lv_s_a, sc_v, sc_gm + sc_s_a))

    return dy_dt
