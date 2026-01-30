
import numpy as np

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
            a           =d['semi major'] * 1e3,
            e           =d['eccentricity'],
            periapsis   =np.radians(d['arg periapsis']),
            long        =np.radians(d['long ascend']),
            i           =np.radians(d['inclination']),
            m0          =np.radians(d['mean anomaly']),
            spr_k       =d['spring k'],
            spr_l0      =d['spring l0'] * 1e-3,
            spr_l1      =d['spring l1'] * 1e-3,
            spr_l2      =d['spring l2'] * 1e-3,
            yaw         =np.radians(d['start yaw']),
            pitch       =np.radians(d['start pitch']),
            mass_lv     =d['mass lv'],
            mass_sc     =d['mass sc'],
            sim_time    =d['sim time']
        )
