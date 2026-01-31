import csv
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist

def read_initial_conditions(filename) -> dict:
    with open(filename, mode='r', newline='') as file:
        ic_dict: dict[str, float] = {}
        csv_reader = csv.reader(file, delimiter=';')
        for row in csv_reader:
            ic_dict.update({ row[0]: float(row[1]) })

    return ic_dict

def draw_bias(axes, point, labels, mult = 1., dcm = None):
    bias_3d = np.identity(3) * mult
    if dcm is not None:
        bias_3d = np.dot(dcm, bias_3d)
    bias_3d += point

    for i, (clr, lbl) in enumerate(zip(('r', 'g', 'b'), labels)):
        axes.plot(*np.column_stack((point, bias_3d[:, i])), color=clr, linewidth=2)
        axes.text(*bias_3d[:, i], lbl, fontsize=12)

def draw_state(axes, rv_vecs):
    if not hasattr(draw_state, 'xyz_lim'):
        draw_state.xyz_lim = float( la.norm(rv_vecs[:3]) / 1e6 )

    # Очистка и настройка графика
    # axes.clear()
    axes.set_xlim([-2 * draw_state.xyz_lim, 2 * draw_state.xyz_lim])
    axes.set_ylim([-2 * draw_state.xyz_lim, 2 * draw_state.xyz_lim])
    axes.set_zlim([-2 * draw_state.xyz_lim, 2 * draw_state.xyz_lim])
    axes.set_box_aspect([1, 1, 1])
    axes.set_xlabel(r'X, km$\times 10^3$')
    axes.set_ylabel(r'Y, km$\times 10^3$')
    axes.set_zlabel(r'Z, km$\times 10^3$')

    vec_r, vec_v = np.hsplit(rv_vecs, 2)

    draw_bias(axes, (0, 0, 0), (r'$\gamma$', r'$y_I$', r'$z_I$'), 1.4 * draw_state.xyz_lim)
    axes.quiver(0, 0, 0, *(vec_r / 1e6 ), color='black')
    axes.quiver(*(vec_r / 1e6 ), *(vec_v / 1e4), color='red')

    plt.draw()

def show_anim(func_draw: Callable, func_arg: list):
    if not hasattr(show_anim, 'fig'):
        show_anim.fig = plt.figure()
    if not hasattr(show_anim, 'axes'):
        show_anim.axes = show_anim.fig.add_subplot(111, projection='3d')

    total_frames = len(func_arg)

    def anim_update(frame) -> Iterable[Artist]:
        func_draw(show_anim.axes, func_arg[frame])
        if frame == total_frames - 1:
            anim.event_source.stop()

    anim = FuncAnimation(
        show_anim.fig, anim_update,
        frames=total_frames,
        interval=50,
        blit=False,
        repeat=False
    )
    plt.show()
