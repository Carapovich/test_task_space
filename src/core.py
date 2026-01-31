
import numpy as np
import sys

from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R


def kepler2eci(a, e, i, long, periapsis, m0) -> NDArray:
    EARTH_FM = 398600.4418e+9

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
    rot_o2i = R.from_euler('xyz', [periapsis, i, long])
    r_i, v_i = rot_o2i.apply([r_o, v_o])
    a_i = - EARTH_FM * r_i / r_c**3

    return np.vstack((r_i, v_i, a_i)).T
