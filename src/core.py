
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

    # Радиус-вектор, вектор линейной скорости и ускорения тела в ИГЦСК
    rot_o2i = R.from_euler('zxz', [periapsis, i, long])
    r_i, v_i = rot_o2i.apply([r_o, v_o])
    a_i = -EARTH_FM * r_i / r_c**3

    return np.hstack((r_i, v_i, a_i))

def motion_equation_rhs(t, y_vecs: NDArray):
    vec_r = y_vecs[:3]
    vec_v = y_vecs[3:]

    g_m = -EARTH_FM * vec_r / la.norm(vec_r)**3

    dy_dt = np.zeros(y_vecs.size)
    dy_dt[:3] = vec_v
    dy_dt[3:] = g_m

    return dy_dt
