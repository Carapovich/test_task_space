
import sys
import numpy as np
import scipy.integrate as integrator

import src.core as core
import src.utils as utils

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

def run_simulation(sim_input: SimulationInput):
    # Задание положения РН по кеплеровым элементам орбиты
    state_lv = core.kepler2eci(sim_input.semi_major, sim_input.eccentricity, sim_input.inclination,
                               sim_input.long_ascend, sim_input.arg_periapsis, sim_input.mean_anomaly)
    # Задание положения КА относительно положения РН
    e_lv2sc, state_sc = core.get_state_relative(state_lv, sim_input.spring_l2,
                                                sim_input.start_yaw, sim_input.start_pitch)

    # Настройка параметров и расчет решения системы ДУ движения
    spring: dict = {"k": sim_input.spring_stiffness, "l0": sim_input.spring_l0, "l1": sim_input.spring_l1}
    t_eval = np.linspace(start=0, stop=sim_input.sim_time, num=int(100 * sim_input.sim_time))
    result = integrator.solve_ivp(core.motion_equation_rhs,
                                  t_span=(0, sim_input.sim_time),
                                  y0=np.vstack((state_lv, state_sc)).ravel(),
                                  t_eval=t_eval,
                                  rtol=1e2*sys.float_info.epsilon,
                                  args=(spring, sim_input.mass_lv, sim_input.mass_sc, e_lv2sc))

    y_vecs = result.y.T.reshape(-1, 4, 3)
    utils.show_anim(func_draw=utils.draw_together, func_arg=list(y_vecs))
